import sys
import os
import json
import sqlite3
import hashlib
import requests
import urllib3
import time
from dotenv import load_dotenv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QMessageBox, QProgressBar, QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, QSettings, Qt

# 禁用 HTTPS 警告（适配Win7旧环境）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 配置常量 =================
# API 配置 (从 .env 文件读取)
load_dotenv()
# 1. Embedding API
EMBEDDING_API_URL = os.getenv("BGE_API_BASE", "https://api.siliconflow.cn/v1") + "/embeddings"
EMBEDDING_API_KEY = os.getenv("BGE_API_KEY", "sk-qeyemnjzmogvhpgsypcyufwidnjcfgjinpwjbnkhwlvhgjrv")
EMBEDDING_MODEL_NAME = os.getenv("BGE_MODEL", "BAAI/bge-m3")

# 2. Rerank API
RERANK_API_URL = os.getenv("BGE_RERANK_API_BASE", "https://api.siliconflow.cn/v1") + "/rerank"
RERANK_API_KEY = os.getenv("BGE_RERANK_API_KEY", "sk-qeyemnjzmogvhpgsypcyufwidnjcfgjinpwjbnkhwlvhgjrv")
RERANK_MODEL_NAME = os.getenv("BGE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# BATCH_SIZE = 8  # 批处理大小，避免一次请求过大

BATCH_SIZE = 8  # 批处理大小，避免一次请求过大

def test_api_connection():
    """测试API连接"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {EMBEDDING_API_KEY}'
    }
    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "input": ["测试文本"],
        "encoding_format": "float"
    }
    
    try:
        response = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, verify=False, timeout=30)
        if response.status_code == 200:
            print("✅ 向量化API连接正常")
            result = response.json()
            if "data" in result:
                print(f"✅ 向量化API响应正常，嵌入维度: {len(result['data'][0]['embedding'])}")
            else:
                print(f"⚠️  向量化API响应格式异常: {result}")
        else:
            print(f"❌ 向量化API连接失败: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"❌ 向量化API连接异常: {str(e)}")
    
    # 测试重排序API
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {RERANK_API_KEY}'
    }
    payload = {
        "model": RERANK_MODEL_NAME,
        "query": "测试查询",
        "documents": ["测试文档"],
        "top_n": 1
    }
    
    try:
        response = requests.post(RERANK_API_URL, headers=headers, json=payload, verify=False, timeout=30)
        if response.status_code == 200:
            print("✅ 重排序API连接正常")
            result = response.json()
            if "results" in result:
                print(f"✅ 重排序API响应正常")
            else:
                print(f"⚠️  重排序API响应格式异常: {result}")
        else:
            print(f"❌ 重排序API连接失败: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"❌ 重排序API连接异常: {str(e)}")


# ================= 样式表 (Dark Mode) =================
STYLESHEET = """
QMainWindow { background-color: #2b2b2b; color: #ffffff; }
QLabel { color: #cccccc; font-size: 14px; font-weight: bold; }
QLineEdit { background-color: #3b3b3b; color: #ffffff; border: 1px solid #555555; padding: 5px; border-radius: 3px; }
QPushButton { background-color: #007acc; color: white; border: none; padding: 8px 15px; border-radius: 4px; font-weight: bold; }
QPushButton:hover { background-color: #005f9e; }
QPushButton:pressed { background-color: #004a80; }
QPushButton:disabled { background-color: #444444; color: #888888; }
QTextEdit { background-color: #1e1e1e; color: #00ff00; border: 1px solid #555555; font-family: Consolas, monospace; font-size: 12px; }
QProgressBar { border: 1px solid #555555; border-radius: 5px; text-align: center; }
QProgressBar::chunk { background-color: #007acc; width: 20px; }
"""

# ================= 工作线程：执行耗时任务 =================
class VectorWorker(QThread):
    log_signal = pyqtSignal(str)       # 发送日志信号
    finish_signal = pyqtSignal(bool, str) # 完成信号
    progress_signal = pyqtSignal(int)  # 进度信号

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path
        self.output_json_path = input_path.replace(".json", "_embedded.json")
        self.output_db_path = input_path.replace(".json", "_rag.db")
        self.batch_size = 8  # 批处理大小，避免一次请求过大

    def generate_stable_id(self, metadata):
        """生成稳定的 ID (doc_title + section_id 的 MD5)"""
        raw_str = f"{metadata.get('doc_title', '')}_{metadata.get('section_id', '')}"
        return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

    def init_db(self):
        """初始化 SQLite 表结构 (符合 Prompt 要求)"""
        self.log_signal.emit(f"正在初始化数据库: {self.output_db_path}")
        conn = sqlite3.connect(self.output_db_path)
        cursor = conn.cursor()
        
        # 1. Vectors 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                embedding TEXT,
                dim INTEGER,
                doc_title TEXT,
                section_id TEXT
            )
        ''')
        
        # 2. Documents 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                embedding_text TEXT,
                section_hint TEXT,
                original_snippet TEXT,
                section_path TEXT,
                depth INTEGER,
                original_length INTEGER
            )
        ''')
        conn.commit()
        return conn

    def call_bge_embedding_api(self, text_batch):
        """调用远程 BGE-M3 嵌入接口"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {EMBEDDING_API_KEY}'
        }
        payload = {
            "model": EMBEDDING_MODEL_NAME,
            "input": text_batch,
            "encoding_format": "float"
        }
        
        try:
            # 使用 verify=False 跳过 SSL 验证 (Win7/内网常见问题)
            response = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, verify=False, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                # 兼容 OpenAI 格式返回
                if "data" in result:
                    return [item["embedding"] for item in result["data"]]
                else:
                    self.log_signal.emit(f"[API Error] 响应格式异常: {result}")
                    return None
            else:
                self.log_signal.emit(f"[API Error] Status: {response.status_code}, Msg: {response.text}")
                return None
        except requests.exceptions.Timeout:
            self.log_signal.emit("[Network Error] 请求超时，请检查网络连接")
            return None
        except requests.exceptions.ConnectionError:
            self.log_signal.emit("[Network Error] 连接错误，请检查网络连接")
            return None
        except Exception as e:
            self.log_signal.emit(f"[Network Error] {str(e)}")
            return None

    def run(self):
        try:
            # 1. 读取 JSON
            self.log_signal.emit(f"正在读取文件: {self.input_path}")
            with open(self.input_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                self.finish_signal.emit(False, "输入 JSON 格式错误，根节点必须是列表。")
                return

            total_items = len(data)
            self.log_signal.emit(f"共加载 {total_items} 条数据，准备开始向量化...")
            
            # 2. 初始化数据库
            conn = self.init_db()
            cursor = conn.cursor()

            processed_results = [] # 用于保存最终 JSON
            
            # 3. 批处理循环
            for i in range(0, total_items, self.batch_size):
                batch_items = data[i : i + self.batch_size]
                batch_texts = [item.get('embedding_text', '') for item in batch_items]
                
                # 过滤空文本
                valid_indices = [idx for idx, txt in enumerate(batch_texts) if txt.strip()]
                valid_texts = [batch_texts[idx] for idx in valid_indices]
                
                if not valid_texts:
                    continue

                self.log_signal.emit(f"正在处理批次: {i+1} - {min(i+self.batch_size, total_items)} / {total_items}")
                
                # 发送请求
                embeddings = self.call_bge_embedding_api(valid_texts)
                
                if embeddings and len(embeddings) == len(valid_texts):
                    # 4. 数据组装与存储
                    for idx_in_batch, vector in zip(valid_indices, embeddings):
                        item = batch_items[idx_in_batch]
                        metadata = item.get('metadata', {})
                        
                        stable_id = self.generate_stable_id(metadata)
                        
                        # 构建完整的存储对象
                        record = {
                            "id": stable_id,
                            "embedding": vector,
                            "embedding_text": item.get('embedding_text', ''),
                            "section_hint": item.get('section_hint', ''),
                            "metadata": metadata,
                            "original_snippet": item.get('original_snippet', '')
                        }
                        
                        processed_results.append(record)
                        
                        # --- 写入 SQLite (事务内) ---
                        # 表 1: Vectors
                        cursor.execute('''
                            INSERT OR REPLACE INTO vectors (id, embedding, dim, doc_title, section_id)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            stable_id,
                            json.dumps(vector), # SQLite存数组通常转为JSON字符串或BLOB
                            len(vector),
                            metadata.get('doc_title', ''),
                            metadata.get('section_id', '')
                        ))
                        
                        # 表 2: Documents
                        cursor.execute('''
                            INSERT OR REPLACE INTO documents (id, embedding_text, section_hint, original_snippet, section_path, depth, original_length)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            stable_id,
                            item.get('embedding_text', ''),
                            item.get('section_hint', ''),
                            item.get('original_snippet', ''),
                            json.dumps(metadata.get('section_path', [])), # 路径转JSON存
                            metadata.get('depth', 0),
                            metadata.get('original_length', 0)
                        ))
                    
                    conn.commit() # 提交当前批次
                else:
                    self.log_signal.emit("❌ 当前批次向量化失败，已跳过。")

                self.progress_signal.emit(int((min(i+self.batch_size, total_items) / total_items) * 100))
                time.sleep(0.5) # 稍微暂停防止速率限制

            conn.close()
            
            # 5. 保存 JSON 结果文件
            self.log_signal.emit(f"正在保存 JSON 结果: {self.output_json_path}")
            with open(self.output_json_path, 'w', encoding='utf-8-sig') as f:
                json.dump(processed_results, f, ensure_ascii=False, indent=2)

            self.finish_signal.emit(True, f"处理完成！\n数据库: {self.output_db_path}\nJSON: {self.output_json_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_signal.emit(f"[Vectorization Error] {str(e)}")
            self.finish_signal.emit(False, str(e))


# ================= 重排序工作线程 =================
class RerankWorker(QThread):
    log_signal = pyqtSignal(str)       # 发送日志信号
    finish_signal = pyqtSignal(bool, str) # 完成信号
    progress_signal = pyqtSignal(int)  # 进度信号

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path
        self.output_json_path = input_path.replace(".json", "_reranked.json")

    def run(self):
        try:
            # 1. 读取 JSON
            self.log_signal.emit(f"正在读取文件: {self.input_path}")
            with open(self.input_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                self.finish_signal.emit(False, "输入 JSON 格式错误，根节点必须是列表。")
                return

            total_items = len(data)
            self.log_signal.emit(f"共加载 {total_items} 条数据，准备开始重排序...")
            
            # 2. 准备重排序数据
            documents = [item.get('embedding_text', '') for item in data]
            query = ""  # 在实际应用中，这里应该是用户的查询
            
            # 3. 调用重排序 API
            if not query:
                # 如果没有提供查询，则从用户输入获取
                query = self.get_user_query_for_rerank()
            
            if not query:
                self.finish_signal.emit(False, "重排序需要查询内容，请提供查询文本")
                return
            
            self.log_signal.emit(f"正在调用重排序 API，处理 {len(documents)} 个文档...")
            rerank_results = self.call_bge_rerank_api(query, documents)
            
            if rerank_results:
                # 4. 根据重排序结果重新排列数据
                reranked_data = []
                for orig_idx, score in rerank_results:
                    if orig_idx < len(data):
                        item = data[orig_idx].copy()
                        item['rerank_score'] = score
                        reranked_data.append(item)
                
                # 5. 保存重排序结果
                self.log_signal.emit(f"正在保存重排序结果: {self.output_json_path}")
                with open(self.output_json_path, 'w', encoding='utf-8-sig') as f:
                    json.dump(reranked_data, f, ensure_ascii=False, indent=2)
                    
                self.finish_signal.emit(True, f"重排序完成！\n输出文件: {self.output_json_path}")
            else:
                self.finish_signal.emit(False, "重排序 API 调用失败")
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_signal.emit(f"[Rerank Error] {str(e)}")
            self.finish_signal.emit(False, str(e))

    def call_bge_rerank_api(self, query, documents):
        """调用远程 BGE 重排序接口"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {RERANK_API_KEY}'
        }
        payload = {
            "model": RERANK_MODEL_NAME,
            "query": query,
            "documents": documents,
            "top_n": len(documents)  # 返回所有文档的排序结果
        }
        
        try:
            response = requests.post(RERANK_API_URL, headers=headers, json=payload, verify=False, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if "results" in result:
                    # 按照返回的索引顺序和相关性分数返回
                    return [(res["index"], res["relevance_score"]) for res in result["results"]]
                else:
                    self.log_signal.emit(f"[Rerank API Error] 响应格式异常: {result}")
                    return None
            else:
                self.log_signal.emit(f"[Rerank API Error] Status: {response.status_code}, Msg: {response.text}")
                return None
        except Exception as e:
            self.log_signal.emit(f"[Rerank Network Error] {str(e)}")
            return None

    def get_user_query_for_rerank(self):
        """获取用户提供的重排序查询"""
        # 这里可以弹出对话框让用户输入查询，但在当前实现中我们暂时返回一个提示
        # 在实际GUI环境中，应该使用QInputDialog.getText
        from PyQt5.QtWidgets import QInputDialog
        query, ok = QInputDialog.getText(None, '重排序查询', '请输入查询内容用于重排序:')
        if ok and query.strip():
            return query.strip()
        return ""


# ================= 主窗体 UI =================
class VectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BGE-M3 Vectorizer & Reranker Client (SiliconFlow API)")
        self.resize(800, 600)
        self.setStyleSheet(STYLESHEET)
        
        # 配置文件路径
        self.settings = QSettings("MyCorp", "BGEClient")
        
        self.initUI()
        self.worker = None

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 1. 标题
        title = QLabel("JSON to BGE-M3 向量化 & 重排序工具")
        title.setStyleSheet("font-size: 18px; color: #007acc;")
        main_layout.addWidget(title)

        # 2. 文件选择区域
        file_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("请选择 Optimized Vector JSON 文件...")
        self.path_input.setReadOnly(True)
        
        btn_browse = QPushButton("📂 选择文件")
        btn_browse.clicked.connect(self.select_file)
        
        file_layout.addWidget(self.path_input)
        file_layout.addWidget(btn_browse)
        main_layout.addLayout(file_layout)

        # 3. 操作选项
        option_layout = QHBoxLayout()
        
        self.operation_combo = QLabel("操作类型:")
        self.operation_combo.setStyleSheet("color: #cccccc; font-size: 14px; font-weight: bold;")
        
        self.operation_selector = QComboBox()
        self.operation_selector.addItems(["向量化 (Embedding)", "重排序 (Rerank)"])
        self.operation_selector.setStyleSheet(
            "QComboBox { background-color: #3b3b3b; color: #ffffff; border: 1px solid #555555; padding: 5px; border-radius: 3px; }"
        )
        
        option_layout.addWidget(self.operation_combo)
        option_layout.addWidget(self.operation_selector)
        
        main_layout.addLayout(option_layout)
        
        # 4. 操作按钮
        self.btn_run = QPushButton("🚀 开始发送 BGE 向量化 & 重排序")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("font-size: 14px;")
        self.btn_run.clicked.connect(self.run_operation)
        main_layout.addWidget(self.btn_run)
        
        # 4. 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # 5. 控制台日志
        main_layout.addWidget(QLabel("系统控制台:"))
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        main_layout.addWidget(self.console)
        
        # 底部状态
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("font-size: 12px; color: #888888;")
        main_layout.addWidget(self.status_label)

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.console.append(f"[{timestamp}] {message}")
        # 滚动到底部
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.End)
        self.console.setTextCursor(cursor)

    def select_file(self):
        # 获取上次保存的目录，默认为桌面
        last_dir = self.settings.value("last_dir", os.path.expanduser("~/Desktop"))
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择 JSON 文件", 
            last_dir, 
            "JSON Files (*.json)"
        )
        
        if file_path:
            self.path_input.setText(file_path)
            # 保存当前选择的目录
            current_dir = os.path.dirname(file_path)
            self.settings.setValue("last_dir", current_dir)
            self.log(f"已加载文件: {file_path}")

    def run_operation(self):
        json_path = self.path_input.text()
        if not json_path or not os.path.exists(json_path):
            QMessageBox.warning(self, "错误", "请先选择有效的 JSON 文件！")
            return
        
        operation_type = self.operation_selector.currentText()
        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.console.clear()
        
        if "重排序" in operation_type:
            self.log("正在启动重排序任务线程...")
            self.worker = RerankWorker(json_path)  # 新的重排序工作线程
        else:
            self.log("正在启动向量化任务线程...")
            self.worker = VectorWorker(json_path)
        
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finish_signal.connect(self.on_task_finished)
        self.worker.start()

    def on_task_finished(self, success, message):
        self.btn_run.setEnabled(True)
        if success:
            QMessageBox.information(self, "完成", message)
            self.log("✅ 任务全部完成")
            self.status_label.setText("任务完成")
        else:
            QMessageBox.critical(self, "失败", f"任务出错: {message}")
            self.log("❌ 任务失败")
            self.status_label.setText("任务失败")

if __name__ == "__main__":
    # 启动前测试API连接
    print("正在测试API连接...")
    test_api_connection()
    print("API连接测试完成，启动GUI...")
    
    app = QApplication(sys.argv)
    window = VectorApp()
    window.show()

    sys.exit(app.exec_())
