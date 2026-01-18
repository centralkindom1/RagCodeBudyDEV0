import sys
import os
import json
import sqlite3
import requests
import urllib3
import time
import numpy as np
import re
import hashlib
import jieba
import jieba.posseg as pseg
from collections import defaultdict
from dotenv import load_dotenv

# PyQt Core 组件用于线程和信号
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

# ================= 配置与环境 =================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''

# API 配置 (从 .env 文件读取)
load_dotenv()
LLM_API_KEY = os.getenv("QWEN_API_KEY", "your qwen api key")
EMBEDDING_API_KEY = os.getenv("BGE_API_KEY", "your bge api key")
RERANK_API_KEY = os.getenv("BGE_RERANK_API_KEY", "sk-nhvzfrxcdohbamqmotmswbfindjdlbfimfbowpmgxprbiaip")

# 1. Embedding API
EMBEDDING_API_URL = os.getenv("BGE_API_BASE", "https://api.siliconflow.cn/v1") + "/embeddings"
EMBEDDING_MODEL_NAME = os.getenv("BGE_MODEL", "BAAI/bge-m3")

# 2. Rerank API
RERANK_API_URL = os.getenv("BGE_RERANK_API_BASE", "https://api.siliconflow.cn/v1") + "/rerank"
RERANK_MODEL_NAME = os.getenv("BGE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# 3. LLM API (使用 Qwen)
LLM_API_URL = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1") + "/chat/completions"
LLM_MODEL_NAME = os.getenv("QWEN_MODEL", "qwen-plus")

# 4. DeepSeek API (备用)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-56d6522248d046cd8f460297d7e4757f")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v3")

# ================= System Prompts =================
REWRITE_SYSTEM_PROMPT = """你是一个工业级 RAG 系统中的「Query Rewrite 模块」。
你的职责不是回答问题，而是：
将用户输入的「简短、模糊或口语化查询」
重写为一个「语义清晰、信息密度高、适合向量检索与 reranker 判断相关性的查询」。

你必须遵守以下原则：
1. 保持用户原始意图不变，不引入不存在的事实
2. 对概念进行合理展开与同义补全（semantic expansion）
3. 输出的查询必须更有利于技术文档、论文、说明性文本的检索
4. 不要输出任何解释、分析或多版本结果
5. 只输出一条重写后的查询文本

重写规则：
- 如果用户查询过短（≤3 个词），必须进行语义扩展
- 如果查询包含歧义词（如 model、train、data、method 等），需根据「技术文档检索」场景进行消歧
- 优先使用完整自然语言描述，而不是关键词堆叠
- 输出长度建议为 1 句话，最多不超过 2 句话"""

DEEPSEEK_R1_BASE_PROMPT = """🎯【角色定义】
你是一个 RAG Final Answer Composer（检索增强生成的最终答案生成器）。
你的任务 不是检索、不是排序、不是猜测，而是：
严格基于已提供的召回结果，对用户问题生成最终、可读、准确的回答。

📥【输入说明】
你将收到一个结构化输入，包含：
1. query: 用户的原始问题
2. retrieved_chunks: 包含来自 [VECTOR] (向量召回) 和 [JSON_Source] (原文硬匹配) 的混合内容。

🔒【强制约束】
1️⃣ 事实来源约束（防幻觉）
❌ 禁止 使用任何外部知识
❌ 禁止 补充未在 retrieved_chunks 中出现的事实
✅ 只允许 基于提供内容进行归纳、重写、总结
如果证据不足：必须明确说明「当前召回内容不足以完整回答该问题」

2️⃣ 内容使用规则（防遗漏）
优先使用 Rank 靠前的内容。
注意区分来源：[JSON_Source] 来源的内容直接来自原始文档，具有最高的事实参考价值。

3️⃣ 噪声处理规则
允许你：修复断行、合并被拆散的句子、去除明显乱码
❌ 不允许“合理猜测”缺失内容

✍️【输出要求】
输出必须满足：
✅ 语言清晰、技术准确
✅ **必须使用 Markdown 格式，包含清晰的段落、列表和加粗**
✅ 不直接大段复制原文（允许短引用）

⚠️【失败兜底策略】
如果所有 retrieved_chunks 与 query 相关性都很弱，必须输出：“根据当前召回的文档内容，无法对该问题给出可靠回答。”"""

# ================= 核心工具函数 =================
def cosine_similarity(vec1, vec2):
    try:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    except Exception:
        return 0.0

def get_text_hash(text):
    """生成文本的 SHA256 哈希，用于严格去重"""
    clean_text = text.strip().lower()
    return hashlib.sha256(clean_text.encode('utf-8')).hexdigest()

def extract_keywords_with_jieba(query, stopwords=None, airline_dict=None, top_n=5):
    """
    【三步走策略应用】
    Step 1: 字典优先 (Airline Dict) -> 权重 3 (最高)
    Step 2: 正则优先 (Flight No/Code) -> 权重 3
    Step 3: Jieba NLP -> 权重 1-2
    """
    if not jieba:
        return query.split() # Fallback
    
    keywords = []
    query_lower = query.lower()
    
    # 预处理 stopwords set
    stop_set = set()
    if stopwords:
        for sw in stopwords:
            stop_set.add(sw.strip().lower())

    # --- Step 1: 字典优先 (Exact Match in Query) ---
    if airline_dict:
        for airline in airline_dict:
            # 简单包含检测
            if airline.lower() in query_lower:
                # 找到航司名，直接作为高权重关键词
                keywords.append((airline, 3))

    # --- Step 2: 正则优先 (Regex for Codes) ---
    # 匹配 CA123, B737, A320 等
    code_pattern = r'[A-Za-z]{2,3}\d{3,4}' 
    codes = re.findall(code_pattern, query)
    for code in codes:
        keywords.append((code, 3))

    # --- Step 3: NLP (Jieba) ---
    words = pseg.cut(query)
    for w in words:
        word = w.word.strip()
        flag = w.flag
        
        if len(word) < 2: continue 
        if word.lower() in stop_set: continue
        
        # 避免重复添加 Step 1/2 已经找到的词
        if any(k[0].lower() == word.lower() for k in keywords):
            continue

        if flag.startswith('n') or flag == 'eng': # 名词或英文
            keywords.append((word, 2)) # 稍低权重
        elif flag.startswith('v'): # 动词
            keywords.append((word, 1)) # 最低权重
        # 其他词性忽略

    # 按权重排序并去重
    keywords.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    result = []
    for k, score in keywords:
        if k not in seen:
            result.append(k)
            seen.add(k)
            
    return result[:top_n]

def is_precise_intent(query):
    """
    动态路由逻辑：检测是否包含大写字母+数字的组合（如 CA1234, B737 等）
    """
    pattern = r'[A-Z]{2,3}\d{3,4}'
    return bool(re.search(pattern, query))

# ================= PageIndex Loader =================
class PageIndexLoader:
    def __init__(self):
        self.index = {}          
        self.ordered_ids = []    
        self.is_loaded = False

    def load_json(self, json_path):
        if not json_path or not os.path.exists(json_path):
            return False, "文件不存在"
        
        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            self.index = {}
            self.ordered_ids = []
            
            root_structure = data.get("structure", []) if isinstance(data, dict) else data
            
            for item in root_structure:
                self._traverse(item, parent_path=[])
            
            self.is_loaded = True
            return True, f"成功加载 PageIndex，包含 {len(self.index)} 个节点"
        except Exception as e:
            return False, f"加载异常: {str(e)}"

    def _traverse(self, node, parent_path):
        current_title = node.get("title", "")
        node_id = str(node.get("node_id", "")) 
        current_path = parent_path + [current_title]
        
        if node_id:
            self.index[node_id] = {
                "title": current_title,
                "text": node.get("text", ""),
                "summary": node.get("summary", ""),
                "path": current_path,
                "raw_node": node 
            }
            self.ordered_ids.append(node_id)
        
        if "nodes" in node and isinstance(node["nodes"], list):
            for child in node["nodes"]:
                self._traverse(child, current_path)

    def get_node(self, node_id):
        return self.index.get(str(node_id))

# ================= Worker: JSON Hard Query (独立线程) =================
class JsonHardQueryWorker(QThread):
    finished_signal = pyqtSignal(list, str) # results, status_msg

    def __init__(self, json_path, keywords):
        super().__init__()
        self.json_path = json_path
        self.keywords = keywords
        self._is_interrupted = False

    def stop(self):
        self._is_interrupted = True

    def run(self):
        if not self.json_path or not os.path.exists(self.json_path) or not self.keywords:
            self.finished_signal.emit([], "JSON 路径无效或无关键词")
            return

        results = []
        try:
            with open(self.json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            if self._is_interrupted: return 

            structure = data.get("structure", []) if isinstance(data, dict) else data
            
            def traverse_search(node, path_stack):
                if self._is_interrupted: return

                current_title = node.get("title", "未命名章节")
                current_text = node.get("text", "")
                node_id = str(node.get("node_id", "unknown"))
                current_path = path_stack + [current_title]
                
                text_lower = current_text.lower()
                hit_count = 0
                for kw in self.keywords:
                    if kw.lower() in text_lower:
                        hit_count += 1
                
                if hit_count > 0:
                    if len(current_text) > 10:
                        path_str = " > ".join(current_path)
                        # 基础分 + 命中奖励
                        score = 10.0 + (hit_count * 2.0)
                        
                        results.append({
                            "id": node_id,
                            "content": current_text,
                            "path": path_str,
                            "score": score,
                            "hit_count": hit_count,
                            "source": "JSON_Source" 
                        })

                if "nodes" in node and isinstance(node["nodes"], list):
                    for child in node["nodes"]:
                        traverse_search(child, current_path)

            for item in structure:
                if self._is_interrupted: break
                traverse_search(item, [])
            
            if self._is_interrupted:
                self.finished_signal.emit([], "JSON 查询已中断")
                return

            results.sort(key=lambda x: x['score'], reverse=True)
            top_results = results[:20] 
            
            self.finished_signal.emit(top_results, f"JSON 原文检索命中: {len(top_results)} 条 (关键词: {self.keywords})")
            
        except Exception as e:
            self.finished_signal.emit([], f"JSON 查询异常: {str(e)}")

# ================= Worker: Recall + RRF Fusion =================
class RecallWorker(QThread):
    log_signal = pyqtSignal(str)          
    result_signal = pyqtSignal(list)      
    summary_signal = pyqtSignal(str)      
    finish_signal = pyqtSignal(bool)      

    def __init__(self, query_text, db_path, json_path, search_mode="smart", summary_model="DeepSeek-R1", 
                 doc_type="不指定类型", stopwords=None, airline_names=None, chunk_limit=40):
        super().__init__()
        self.original_query = query_text 
        self.search_query = query_text   
        self.db_path = db_path
        self.json_path = json_path
        self.search_mode = search_mode 
        self.summary_model = summary_model 
        self.doc_type = doc_type 
        self.stopwords = stopwords if stopwords else []
        self.airline_names = airline_names if airline_names else [] # Step 1 Dictionary
        self.chunk_limit = chunk_limit # 动态流控参数
        
        self.page_index = PageIndexLoader()
        self.json_search_results = [] 
        
        self._is_interrupted = False
        self._json_worker = None

    def stop(self):
        """外部调用以停止任务"""
        self.log("🛑 收到停止指令，正在中断任务...")
        self._is_interrupted = True
        if self._json_worker:
            self._json_worker.stop()
            self._json_worker.wait(100) 

    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.log_signal.emit(f"[{timestamp}] {msg}")

    def on_json_search_finished(self, results, msg):
        self.json_search_results = results
        self.log(f"📄 {msg}")

    # --- Step 0: Query Rewrite ---
    def rewrite_query(self, original_query):
        if self._is_interrupted: return None

        if self.search_mode == "precise":
            self.log("⏩ 精准模式：跳过查询重写")
            return original_query

        self.log(f"🧠 正在请求 DeepSeek-V3 进行语义重写 (类型偏好: {self.doc_type})...")
        
        doc_type_hint = ""
        if self.doc_type and self.doc_type != "不指定类型":
            doc_type_hint = f"\n\n[Important Context]: The user explicitly expects content from document type: '{self.doc_type}'. Please refine the query to imply this context."

        headers = { 'Content-Type': 'application/json', 'Authorization': f'Bearer {LLM_API_KEY}' }
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"用户查询：\n{original_query}{doc_type_hint}\n\n请输出重写后的查询："}
        ]
        payload = {
            "model": LLM_MODEL_NAME,
            "messages": messages,
            "temperature": 0.7,
            "stream": False
        }

        try:
            start_time = time.time()
            if self._is_interrupted: return None

            response = requests.post(LLM_API_URL, headers=headers, json=payload, verify=False, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content'].strip()
                content = content.replace('"', '').replace("'", "")
                
                self.log(f"✅ Rewrite 完成 ({time.time() - start_time:.2f}s)")
                self.log(f"   Original: {original_query}")
                self.log(f"   Rewritten: {content}")
                return content
            else:
                self.log(f"⚠️ Rewrite API 返回错误: {response.status_code}，将使用原始查询。")
                return original_query
        except Exception as e:
            self.log(f"⚠️ Rewrite 调用异常: {str(e)}，将使用原始查询。")
            return original_query

    # --- Step 1: Embedding ---
    def get_remote_embedding(self, text):
        if self._is_interrupted: return None
        self.log(f"📡 正在计算向量 Embedding: {text[:30]}...")
        headers = { 'Content-Type': 'application/json', 'Authorization': f'Bearer {EMBEDDING_API_KEY}' }
        payload = { "model": EMBEDDING_MODEL_NAME, "input": [text] }
        
        try:
            response = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, verify=False, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    return data['data'][0]['embedding']
        except Exception as e:
            self.log(f"❌ Embedding 网络异常: {str(e)}")
        return None

    # --- Step 2: Rerank API ---
    def rerank_with_bge(self, query, candidates_text_list):
        if not candidates_text_list or self._is_interrupted:
            return []
        
        rerank_query = query
        if self.doc_type and self.doc_type != "不指定类型":
            rerank_query = f"{query} (Prefer document type: {self.doc_type})"
            self.log(f"⚖️ Reranker 使用增强 Query: {rerank_query}")

        self.log(f"⚖️ Reranker ({RERANK_MODEL_NAME}) 正在重排 {len(candidates_text_list)} 条数据...")
        headers = { 'Content-Type': 'application/json', 'Authorization': f'Bearer {RERANK_API_KEY}' }
        
        payload = {
            "model": RERANK_MODEL_NAME,
            "query": rerank_query, 
            "documents": candidates_text_list 
        }

        try:
            start_time = time.time()
            if self._is_interrupted: return None
            
            response = requests.post(RERANK_API_URL, headers=headers, json=payload, verify=False, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                scores = [0.0] * len(candidates_text_list)
                
                if "results" in data:
                    for res in data["results"]:
                        idx = res.get("index")
                        score = res.get("relevance_score", 0.0)
                        if idx is not None and 0 <= idx < len(scores):
                            scores[idx] = score
                elif isinstance(data, list):
                     scores = data

                self.log(f"✅ Reranker 完成，耗时: {time.time() - start_time:.2f}s")
                return scores
            else:
                self.log(f"⚠️ Reranker 请求失败: {response.status_code}")
                return None
        except Exception as e:
            self.log(f"⚠️ Reranker 调用异常: {str(e)}")
            return None

    # --- Step 4: LLM Summary ---
    def call_deepseek_summary(self, user_original_query, top_results):
        if self._is_interrupted: return

        target_model = self.summary_model 
        self.log(f"🧠 正在请求 {target_model} 生成最终回答 (Stream=True)...")
        self.summary_signal.emit(f"> 🚀 **{target_model} 已连接，准备生成...**\n\n")

        current_system_prompt = DEEPSEEK_R1_BASE_PROMPT
        if self.doc_type and self.doc_type != "不指定类型":
            doc_type_constraint = f"""
\n⚠️【文档类型强制偏好】
用户期望的答案主要来自文档类型：【{self.doc_type}】。
1. 回答时请优先参考该类型的内容。
2. 但如果跨类型内容明显有助于回答问题，请合理补充。
"""
            current_system_prompt += doc_type_constraint

        context_str = ""
        for item in top_results:
            source_tag = item.get('source', 'VECTOR')
            context_str += f"""
---
[Rank {item['rank']}] [Source: {source_tag}] (RRF: {item['final_score']:.4f})
Section Path: {item['path']}
Content:
{item['content']}
"""
        
        user_prompt_content = f"Query: {user_original_query}\n\nRetrieved Chunks:{context_str}"

        # 模型名称映射：将前端显示名称转换为API实际模型名称
        model_mapping = {
            "qwen-plus": "qwen-plus",
            "qwen-turbo": "qwen-turbo",
            "qwen-max": "qwen-max",
            "DeepSeek-R1": "deepseek-chat",
            "DeepSeek-V3": "deepseek-v3",
            "X1-70B-thinking": "x1-70b-thinking",
            "X1-70B-fast": "x1-70b-fast"
        }
        
        # 根据目标模型选择适当的API配置
        if target_model in ["DeepSeek-R1", "DeepSeek-V3"]:
            api_key = DEEPSEEK_API_KEY
            api_url = f"{DEEPSEEK_API_BASE}/chat/completions"
            # 映射到实际的API模型名称
            actual_model = model_mapping.get(target_model, target_model)
        elif target_model.startswith("qwen"):
            api_key = LLM_API_KEY
            api_url = LLM_API_URL
            # 映射到实际的API模型名称
            actual_model = model_mapping.get(target_model, target_model)
        else:
            api_key = LLM_API_KEY
            api_url = LLM_API_URL
            actual_model = target_model
        
        headers = { 'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}' }
        payload = {
            "model": actual_model, 
            "messages": [
                {"role": "system", "content": current_system_prompt},
                {"role": "user", "content": user_prompt_content}
            ],
            "stream": True,
            "temperature": 0.6
        }

        try:
            if self._is_interrupted: return

            response = requests.post(api_url, headers=headers, json=payload, verify=False, stream=True, timeout=120)
            
            if response.status_code == 200:
                full_reasoning = ""
                full_content = ""
                is_thinking_logged = False
                
                for line in response.iter_lines():
                    if self._is_interrupted: 
                        self.log("🛑 总结生成已中断")
                        self.summary_signal.emit("\n\n[用户终止了生成]")
                        break

                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:] 
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                json_chunk = json.loads(data_str)
                                delta = json_chunk['choices'][0]['delta']
                                
                                current_reasoning_delta = delta.get('reasoning_content', '')
                                current_content_delta = delta.get('content', '')
                                updated = False
                                
                                if current_reasoning_delta:
                                    if not is_thinking_logged:
                                        self.log("🧠 检测到思维链 (CoT)，正在思考...")
                                        is_thinking_logged = True
                                    full_reasoning += current_reasoning_delta
                                    updated = True
                                
                                if current_content_delta:
                                    full_content += current_content_delta
                                    updated = True

                                if updated:
                                    formatted_output = ""
                                    if full_reasoning:
                                        clean_reasoning = full_reasoning.replace('\n', '\n> ')
                                        formatted_output += f"> 🧠 **Thinking Process:**\n> {clean_reasoning}\n\n"
                                    
                                    if full_reasoning and full_content:
                                        formatted_output += "---\n\n" 
                                        
                                    if full_content:
                                        formatted_output += f"{full_content}"
                                        
                                    self.summary_signal.emit(formatted_output)
                            except Exception:
                                continue
                if not self._is_interrupted:
                    self.log(f"✅ {target_model} 总结生成完毕")
            else:
                self.log(f"❌ API 错误: {response.status_code}")
                self.summary_signal.emit(f"⚠️ 无法生成总结: API Error {response.status_code}")

        except Exception as e:
            self.log(f"❌ DeepSeek 调用异常: {str(e)}")
            self.summary_signal.emit(f"⚠️ 总结生成失败: {str(e)}")

    # --- 核心算法: RRF Fusion ---
    def apply_rrf_fusion(self, vector_items, json_items, k=60):
        fused_scores = defaultdict(float)
        item_map = {}
        
        # 1. Vector Results
        for rank, item in enumerate(vector_items):
            doc_id = item['id']
            item_map[doc_id] = item
            fused_scores[doc_id] += 1.0 / (k + rank + 1)
            
        # 2. JSON Results (Boost Logic)
        is_precise = is_precise_intent(self.original_query)
        json_boost = 1.0
        
        # --- 策略调整: 书籍模式下降低 JSON 关键词权重 ---
        is_book_mode = self.doc_type in ["书籍/教材", "长篇论文"]
        
        if self.search_mode == 'precise':
            json_boost = 5.0 
        elif self.search_mode == 'smart':
            if is_book_mode:
                self.log("📚 检测到书籍/长文档模式：主动降低关键词权重，优先语义召回")
                json_boost = 0.5  # 降权，防止书籍中非定义的关键词干扰
            elif is_precise:
                self.log("💡 动态路由: 检测到精确代码/航班号，自动提升 JSON 权重")
                json_boost = 3.0 
            else:
                json_boost = 1.0
        elif self.search_mode == 'fuzzy':
            json_boost = 0.5 

        for rank, item in enumerate(json_items):
            doc_id = item['id']
            if doc_id not in item_map:
                item_map[doc_id] = item
                item_map[doc_id]['debug_score'] = "JSON_New"
            
            fused_scores[doc_id] += json_boost * (1.0 / (k + rank + 1))
            if "JSON" not in item_map[doc_id].get('source', ''):
                item_map[doc_id]['source'] = "MIXED (Vec+JSON)"

        # 3. Sort
        sorted_doc_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        # 4. Fingerprint Deduplication
        final_results = []
        seen_fingerprints = set()
        
        for doc_id in sorted_doc_ids:
            item = item_map[doc_id]
            item['final_score'] = fused_scores[doc_id]
            
            content_fingerprint = get_text_hash(item.get('content', ''))
            
            if content_fingerprint in seen_fingerprints:
                continue
            
            seen_fingerprints.add(content_fingerprint)
            final_results.append(item)
            
        return final_results

    def run(self):
        try:
            self._is_interrupted = False

            # 0. Load PageIndex
            has_pageindex = False
            if self.json_path:
                self.log(f"加载 PageIndex: {os.path.basename(self.json_path)}...")
                success, msg = self.page_index.load_json(self.json_path)
                if success:
                    has_pageindex = True
                else:
                    self.log(f"⚠️ PageIndex 加载失败: {msg}")
            
            if self._is_interrupted: return

            # --- Concurrent Step: JSON Hard Query (Three-Step Enabled) ---
            self._json_worker = None
            if self.json_path and self.search_mode != 'fuzzy': 
                # 【Update】传入 airline_names 和 stopwords
                keywords = extract_keywords_with_jieba(
                    self.original_query, 
                    self.stopwords, 
                    self.airline_names
                )
                self.log(f"🔍 [三步走策略] 提取关键词: {keywords}")
                
                if keywords:
                    self.log("🚀 启动 JSON 原文硬查询线程...")
                    self._json_worker = JsonHardQueryWorker(self.json_path, keywords)
                    self._json_worker.finished_signal.connect(self.on_json_search_finished)
                    self._json_worker.start()
            
            if self._is_interrupted: return

            # --- Step 0: Query Rewrite ---
            rewritten = self.rewrite_query(self.original_query)
            if self._is_interrupted: return
            self.search_query = rewritten if rewritten else self.original_query

            # --- Step 1: Query Vector ---
            vector_candidates = []
            
            query_vec_list = self.get_remote_embedding(self.search_query)
            if self._is_interrupted: return

            if query_vec_list and os.path.exists(self.db_path):
                query_vec_np = np.array(query_vec_list, dtype=np.float32)

                self.log(f"📂 正在连接数据库: {os.path.basename(self.db_path)}")
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT id, embedding, section_id FROM vectors")
                rows = cursor.fetchall()
                
                raw_candidates = []
                for row in rows:
                    if self._is_interrupted: break 
                    v_id, emb_json, sec_id_db = row
                    try:
                        doc_vec = np.array(json.loads(emb_json), dtype=np.float32)
                        score = cosine_similarity(query_vec_np, doc_vec)
                        raw_candidates.append({"id": v_id, "vec_score": score, "section_id": str(sec_id_db)})
                    except: continue

                if self._is_interrupted: 
                    conn.close()
                    return

                raw_candidates.sort(key=lambda x: x["vec_score"], reverse=True)
                
                # 【流控关键点】使用 chunk_limit 进行截断，防止 Reranker 过载
                current_limit = self.chunk_limit
                self.log(f"⚡ [流控] 限制向量召回数量为: {current_limit} (Mode: {self.doc_type})")
                top_candidates_raw = raw_candidates[:current_limit] 
                
                rerank_input_texts = []
                
                for item in top_candidates_raw:
                    if self._is_interrupted: break
                    sec_id = item["section_id"]
                    node_info = self.page_index.get_node(sec_id) if has_pageindex else None
                    
                    raw_text = ""
                    path_str = ""
                    summary_text = ""

                    if node_info:
                        raw_text = node_info['text']
                        path_str = " > ".join(node_info['path'])
                        summary_text = node_info.get('summary', '')
                    else:
                        cursor.execute("SELECT embedding_text, original_snippet, section_path FROM documents WHERE id=?", (sec_id,))
                        db_row = cursor.fetchone()
                        if db_row:
                            emb_summary = db_row[0] if db_row[0] else ""
                            raw_detail = db_row[1] if db_row[1] else ""
                            raw_text = f"【内容摘要】：{emb_summary}\n\n【原始数据】：{raw_detail}"
                            path_str = str(db_row[2])
                        else:
                            continue

                    rerank_input_texts.append(f"Section Path: {path_str}\nContent: {raw_text}")
                    display_content = f"[Summary]\n{summary_text}\n\n[Text]\n{raw_text}" if summary_text else raw_text
                    
                    vector_candidates.append({
                        "id": sec_id, 
                        "vec_score": item["vec_score"],
                        "path": path_str,
                        "content": display_content,
                        "source": "VECTOR" 
                    })
                
                conn.close()
                
                if not self._is_interrupted and vector_candidates:
                    rerank_scores = self.rerank_with_bge(self.search_query, rerank_input_texts)
                    if rerank_scores:
                        for idx, candidate in enumerate(vector_candidates):
                            candidate['rerank_score'] = rerank_scores[idx]
                        
                        vector_candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
                        self.log(f"✅ Vector 通道准备就绪: {len(vector_candidates)} 条 (已 Rerank)")

            # --- Wait for JSON Search ---
            json_results = []
            if self._json_worker:
                self.log("⏳ 等待 JSON 原文硬查询线程完成...")
                while self._json_worker.isRunning():
                    if self._is_interrupted:
                        self._json_worker.stop()
                        break
                    self._json_worker.wait(100) 

                json_results = self.json_search_results
            
            if self._is_interrupted:
                self.finish_signal.emit(False)
                return

            # --- RRF Fusion ---
            self.log("⚖️ 执行 RRF 融合与内容指纹去重...")
            final_top_results = self.apply_rrf_fusion(vector_candidates, json_results)
            
            final_top_results = final_top_results[:12]
            self.log(f"✅ 最终召回: {len(final_top_results)} 条唯一内容")
            
            for idx, res in enumerate(final_top_results):
                res['rank'] = idx + 1
                
            self.result_signal.emit(final_top_results)
            
            # --- LLM Summary ---
            self.call_deepseek_summary(self.original_query, final_top_results)
            
            if self._is_interrupted:
                self.finish_signal.emit(False)
            else:
                self.finish_signal.emit(True)

        except Exception as e:
            self.log(f"❌ 严重错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

            self.finish_signal.emit(False)
