#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API Configuration Test Script
测试所有API密钥和端点是否配置正确
"""
import sys
import os
import io
import requests
import urllib3
from dotenv import load_dotenv

# 修复Windows控制台编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 加载环境变量
load_dotenv()

print("=" * 60)
print("RAG系统 API配置测试")
print("=" * 60)

# 测试函数
def test_qwen_api():
    """测试Qwen API"""
    print("\n[1/3] 测试 Qwen API (LLM)...")
    api_key = os.getenv("QWEN_API_KEY")
    api_base = os.getenv("QWEN_API_BASE")
    model = os.getenv("QWEN_MODEL")

    print(f"  API Key: {api_key[:20]}...{api_key[-5:] if api_key else 'None'}")
    print(f"  API Base: {api_base}")
    print(f"  Model: {model}")

    if not api_key:
        print("  ❌ 未配置 QWEN_API_KEY")
        return False

    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10, verify=False)
        print(f"  Status Code: {response.status_code}")

        if response.status_code == 200:
            print("  ✅ Qwen API 连接成功!")
            result = response.json()
            print(f"  Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')}")
            return True
        else:
            print(f"  ❌ 请求失败: {response.text}")
            return False
    except Exception as e:
        print(f"  ❌ 异常: {str(e)}")
        return False

def test_bge_embedding_api():
    """测试BGE Embedding API"""
    print("\n[2/3] 测试 BGE Embedding API...")
    api_key = os.getenv("BGE_API_KEY")
    api_base = os.getenv("BGE_API_BASE")
    model = os.getenv("BGE_MODEL")

    print(f"  API Key: {api_key[:20]}...{api_key[-5:] if api_key else 'None'}")
    print(f"  API Base: {api_base}")
    print(f"  Model: {model}")

    if not api_key:
        print("  ❌ 未配置 BGE_API_KEY")
        return False

    url = f"{api_base}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "input": ["测试文本"]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10, verify=False)
        print(f"  Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                embedding = result['data'][0]['embedding']
                print(f"  ✅ BGE Embedding API 连接成功!")
                print(f"  Embedding 维度: {len(embedding)}")
                return True
            else:
                print(f"  ❌ 响应格式异常: {result}")
                return False
        else:
            print(f"  ❌ 请求失败: {response.text}")
            return False
    except Exception as e:
        print(f"  ❌ 异常: {str(e)}")
        return False

def test_bge_rerank_api():
    """测试BGE Rerank API"""
    print("\n[3/3] 测试 BGE Rerank API...")
    api_key = os.getenv("BGE_RERANK_API_KEY")
    api_base = os.getenv("BGE_RERANK_API_BASE")
    model = os.getenv("BGE_RERANK_MODEL")

    print(f"  API Key: {api_key[:20]}...{api_key[-5:] if api_key else 'None'}")
    print(f"  API Base: {api_base}")
    print(f"  Model: {model}")

    if not api_key:
        print("  ❌ 未配置 BGE_RERANK_API_KEY")
        return False

    url = f"{api_base}/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "query": "测试查询",
        "documents": ["文档1内容", "文档2内容", "文档3内容"]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10, verify=False)
        print(f"  Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if 'results' in result:
                print(f"  ✅ BGE Rerank API 连接成功!")
                print(f"  重排结果数: {len(result['results'])}")
                return True
            else:
                print(f"  ❌ 响应格式异常: {result}")
                return False
        else:
            print(f"  ❌ 请求失败: {response.text}")
            return False
    except Exception as e:
        print(f"  ❌ 异常: {str(e)}")
        return False

# 主测试流程
if __name__ == "__main__":
    results = []

    results.append(("Qwen LLM", test_qwen_api()))
    results.append(("BGE Embedding", test_bge_embedding_api()))
    results.append(("BGE Rerank", test_bge_rerank_api()))

    # 打印测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)

    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name:20s} : {status}")

    all_passed = all(success for _, success in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有API测试通过! 系统已就绪。")
    else:
        print("⚠️ 部分API测试失败，请检查配置。")
    print("=" * 60)
