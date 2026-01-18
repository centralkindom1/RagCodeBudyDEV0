#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rule-Based Tree-to-Flat JSON Converter for RAG Vectorization
This script converts tree-structured PageIndex JSON to flat vector-ready JSON
without using LLMs, using rule-based logic for fast conversion.
"""

import json
import os
import sys
import argparse
import time
from typing import Dict, List, Any


def log(msg: str, level: str = "INFO"):
    """Standard logging function compatible with frontend worker."""
    print(f"[{level}] {msg}", flush=True)


def recursive_walk(nodes: List[Dict], path: List[str] = [], depth: int = 1):
    """
    Recursively walk through tree structure and yield flattened items.
    
    Args:
        nodes: List of tree nodes to walk
        path: Current path in the tree
        depth: Current depth level
    
    Yields:
        Dictionary containing node, path and depth
    """
    for node in nodes:
        current_title = node.get("title", "Untitled")
        current_title = current_title.replace('\n', ' ').strip()
        current_path = path + [current_title]
        
        yield {"node": node, "path": current_path, "depth": depth}
        
        # Process child nodes if they exist
        if "nodes" in node and isinstance(node["nodes"], list):
            yield from recursive_walk(node["nodes"], current_path, depth + 1)


def convert_tree_to_flat_json(input_path: str, output_path: str):
    """
    Convert tree-structured JSON to flat vector-ready JSON using rule-based logic.
    
    Args:
        input_path: Path to input tree-structured JSON
        output_path: Path to output flat JSON
    """
    log(f"Starting rule-based conversion: {input_path} -> {output_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        log(f"Input file not found: {input_path}", "ERROR")
        return False
    
    try:
        # Read input JSON
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # Determine input structure
        root_nodes = []
        doc_title = "Unknown Document"
        
        if isinstance(data, list):
            root_nodes = data
        elif isinstance(data, dict):
            root_nodes = data.get("structure", [])
            doc_title = data.get("doc_name", data.get("title", "Unknown Document"))
        
        # Prepare output data structure
        output_data = []
        processed_count = 0
        
        # Walk through the tree structure and convert to flat format
        for item in recursive_walk(root_nodes):
            node = item["node"]
            path = item["path"]
            depth = item["depth"]
            content = node.get("text", node.get("content", ""))
            node_id = node.get("node_id", f"node_{processed_count:04d}")
            
            # Apply filtering logic similar to TAB B
            has_children = "nodes" in node and isinstance(node["nodes"], list) and len(node["nodes"]) > 0
            
            # Skip if content is too short and has no children
            if (not content or len(content.strip()) < 10) and not has_children:
                continue
            
            # Build path string for metadata
            path_str = " > ".join(path)
            
            # Create semantic intro based on path and content
            if has_children and not content:
                semantic_intro = f"This is section '{path_str}' from document '{doc_title}' containing sub-sections."
            elif content:
                semantic_intro = f"This is section '{path_str}' from document '{doc_title}', containing {len(content)} characters of text content."
            else:
                semantic_intro = f"This is section '{path_str}' from document '{doc_title}'."
            
            # Determine section hint based on path and content characteristics
            section_hint = "General"
            if "table" in content.lower() or "chart" in content.lower():
                section_hint = "Data Table/Chart"
            elif "procedure" in content.lower() or "process" in content.lower():
                section_hint = "Procedure/Process"
            elif "specification" in content.lower() or "spec" in content.lower():
                section_hint = "Specification"
            elif "warning" in content.lower() or "caution" in content.lower() or "danger" in content.lower():
                section_hint = "Safety Warning"
            elif "figure" in content.lower() or "image" in content.lower():
                section_hint = "Figure Reference"
            
            # Build final text for embedding - similar to TAB B strategy 0 (lossless)
            if not content and has_children:
                final_text = f"{semantic_intro}\n(Contains child sections)"
            else:
                final_text = f"{semantic_intro}\n\n[Original Content]:\n{content}"
            
            # Create the output item in the same format as TAB B
            final_item = {
                "embedding_text": final_text,
                "section_hint": section_hint,
                "metadata": {
                    "doc_title": doc_title,
                    "section_id": node_id,
                    "section_path": path,
                    "depth": depth,
                    "original_length": len(content),
                    "strategy": 0  # Using strategy 0 (lossless) by default
                },
                "original_snippet": content
            }
            
            output_data.append(final_item)
            processed_count += 1
            
            # Emit progress update
            print(f"@@PROGRESS@@{{\"phase\": \"Converting\", \"current\": {processed_count}, \"total\": 0}}", flush=True)
            
            # Small delay to allow for UI updates
            time.sleep(0.01)
        
        # Write output JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        log(f"Conversion Complete! Processed: {processed_count} items")
        log(f"Output: {output_path}")
        return True
        
    except Exception as e:
        log(f"Critical Error during conversion: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to handle command-line arguments and run conversion."""
    parser = argparse.ArgumentParser(description="Rule-based Tree-to-Flat JSON Converter for RAG Vectorization")
    parser.add_argument("input", help="Input JSON file path (tree structure)")
    parser.add_argument("output", help="Output JSON file path (flat structure)")
    
    args = parser.parse_args()
    
    # Ensure UTF-8 encoding for output
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    success = convert_tree_to_flat_json(args.input, args.output)
    
    if success:
        log("Rule-based conversion completed successfully!", "SUCCESS")
    else:
        log("Rule-based conversion failed!", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()