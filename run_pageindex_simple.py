#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版PageIndex - 跳过TOC检测,直接使用文档结构
适用于: TOC检测失败或API连接问题的场景
"""
import argparse
import os
import sys
import json
import asyncio

# UTF-8 编码设置
if sys.stdout:
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr:
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pageindex.utils import config, ConfigLoader
from pageindex.page_index import page_index_main

def main():
    parser = argparse.ArgumentParser(description="PageIndex Simple - No TOC Detection")
    parser.add_argument('--pdf_path', type=str, required=True, help="Path to PDF file")
    parser.add_argument('--model', type=str, default="qwen-plus", help="AI Model to use")

    args = parser.parse_args()

    print(f"[INFO] Starting SIMPLE indexing (NO TOC) for: {args.pdf_path}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] 此版本跳过TOC检测,直接使用文档结构...")

    try:
        # 创建配置 - 禁用TOC检测
        opt = config(
            pdf_path=args.pdf_path,
            model=args.model,
            toc_check_page_num=0,  # 禁用TOC检测 (关键修改)
            max_page_num_each_node=10,
            max_token_num_each_node=5000,
            if_add_node_id='yes',
            if_add_node_text='yes',
            if_add_node_summary='yes',
            if_add_doc_description='yes'
        )

        print("[INFO] 开始处理PDF...")
        result = page_index_main(doc=args.pdf_path, opt=opt)

        if result:
            print(f"\n[SUCCESS] 处理完成!")
            print(f"[INFO] 文档名: {result.get('doc_name', 'Unknown')}")
            print(f"[INFO] 章节数量: {len(result.get('structure', []))}")

            # 保存到logs目录
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
            output_filename = f"{pdf_name}_{timestamp}.json"
            output_path = os.path.join("logs", output_filename)

            os.makedirs("logs", exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"[INFO] 结果已保存至: {output_path}")
            return True
        else:
            print("[ERROR] 处理失败,结果为空")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to process PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
