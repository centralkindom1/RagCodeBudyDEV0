#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单PDF解析测试
测试PDF文件是否能正常解析,不调用LLM API
"""
import sys
import os
import json
from io import BytesIO

# 设置UTF-8编码
if sys.stdout:
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr:
    sys.stderr.reconfigure(encoding='utf-8')

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import pdfplumber
    print("✅ pdfplumber 已加载")
except ImportError:
    print("❌ pdfplumber 未安装,尝试使用 PyPDF2")
    import PyPDF2

def test_pdf_parsing(pdf_path):
    """测试PDF文件解析"""
    print(f"\n{'='*60}")
    print(f"测试PDF解析: {pdf_path}")
    print('='*60)

    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        return False

    try:
        # 尝试使用 pdfplumber
        if 'pdfplumber' in sys.modules:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"✅ PDF打开成功,共 {total_pages} 页")

                # 提取前3页内容
                for i in range(min(3, total_pages)):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    print(f"\n--- 第 {i+1} 页预览 ---")
                    print(text[:200] if text else "(空白页)")

                return True
        else:
            # 使用 PyPDF2 作为备选
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)
                print(f"✅ PDF打开成功,共 {total_pages} 页")

                for i in range(min(3, total_pages)):
                    page = pdf_reader.pages[i]
                    text = page.extract_text()
                    print(f"\n--- 第 {i+1} 页预览 ---")
                    print(text[:200] if text else "(空白页)")

                return True

    except Exception as e:
        print(f"❌ PDF解析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 测试文件列表
    test_files = [
        "D:/RagCodeBudyDEV0/tests/pdfs/PRML.pdf",
        "D:/RagCodeBudyDEV0/tests/pdfs/earthmover.pdf"
    ]

    for pdf_file in test_files:
        success = test_pdf_parsing(pdf_file)
        if success:
            print(f"\n✅ {os.path.basename(pdf_file)} 解析测试通过")
        else:
            print(f"\n❌ {os.path.basename(pdf_file)} 解析测试失败")
        print()
