"""命令行界面模块"""

import asyncio
import sys
import json
import shlex
import traceback
from typing import List, Dict, Any, Optional, Tuple

from .utils import parse_json_params, parse_server_and_name, print_help_message


async def run_interactive_cli(client):
    """运行交互式命令行界面
    
    Args:
        client: GenericMCPClient实例
    """
    print("\n欢迎使用LiteMCP客户端。输入 'help' 查看可用命令，'exit' 退出。")
    print("提示: 首先使用 'connect <服务器名>' 连接到服务器。\n")
    
    try:
        while True:
            # 显示提示符
            current_server = ""
            if client.current_connection:
                current_server = f"[{client.current_connection.name}]"
            
            try:
                user_input = input(f"LiteMCP{current_server}> ").strip()
            except EOFError:
                # 处理Ctrl+D
                print("\n已退出。")
                break
                
            if not user_input:
                continue  # 忽略空行
                
            # 解析命令和参数
            parts = shlex.split(user_input)
            command = parts[0].lower()
            args = parts[1:]
            
            try:
                # 执行命令
                await process_command(client, command, args, user_input)
            except Exception as e:
                print(f"执行命令时出错: {str(e)}")
                traceback.print_exc()
                
    except KeyboardInterrupt:
        print("\n已中断。")
    finally:
        print("退出交互模式...")

async def process_command(client, command: str, args: List[str], raw_input: str):
    """处理单个命令
    
    Args:
        client: GenericMCPClient实例
        command: 命令名称
        args: 命令参数
        raw_input: 原始用户输入
    """
    # 处理退出命令
    if command in ["exit", "quit"]:
        print("正在退出...")
        raise KeyboardInterrupt()  # 使用 KeyboardInterrupt 退出主循环
        
    # 处理帮助命令
    elif command in ["help", "?"]:
        print_help_message()
        
    # 处理连接命令
    elif command == "connect":
        if not args:
            print("错误: 请指定服务器名称。")
            print("用法: connect <服务器名>")
            return
            
        server_name = args[0]
        await client.connect_to_server_by_name(server_name)
        
    # 连接所有默认服务器
    elif command == "connect-all":
        await client.connect_all_default_servers()
        
    # 断开连接
    elif command == "disconnect":
        if not args:
            print("错误: 请指定服务器名称。")
            print("用法: disconnect <服务器名>")
            return
            
        server_name = args[0]
        await client.disconnect_from_server(server_name)
        
    # 切换当前连接
    elif command == "switch":
        if not args:
            print("错误: 请指定服务器名称。")
            print("用法: switch <服务器名>")
            return
            
        server_name = args[0]
        client.switch_current_connection(server_name)
        
    # 列出连接
    elif command in ["connections", "conn"]:
        client.list_connections()
        
    # 列出工具
    elif command == "tools":
        client.list_all_tools()
        
    # 列出资源
    elif command in ["resources", "res"]:
        client.list_all_resources()
        
    # 列出提示模板
    elif command == "prompts":
        client.list_all_prompts()
        
    # 调用工具
    elif command == "call":
        if not args:
            print("错误: 请指定工具。")
            print("用法: call <服务器名.工具名> [JSON参数]")
            return
            
        # 解析工具规格
        try:
            server_name, tool_name = parse_server_and_name(args[0])
        except ValueError as e:
            print(f"错误: {str(e)}")
            return
            
        # 解析参数
        params = {}
        if len(args) > 1:
            params_str = " ".join(args[1:])
            try:
                params = parse_json_params(params_str)
            except ValueError as e:
                print(f"错误: {str(e)}")
                return
                
        try:
            result = await client.call_tool_by_server(server_name, tool_name, params)
            print("\n调用结果:")
            # 直接打印结果，不尝试 JSON 序列化
            print(result)
        except Exception as e:
            print(f"工具调用失败: {str(e)}")
            
    # 获取资源
    elif command == "get":
        if not args:
            print("错误: 请指定资源URI。")
            print("用法: get <服务器名.资源URI>")
            return
            
        # 解析资源规格
        try:
            server_name, resource_uri = parse_server_and_name(args[0])
        except ValueError as e:
            print(f"错误: {str(e)}")
            return
            
        try:
            result = await client.get_resource_by_server(server_name, resource_uri)
            print("\n资源内容:")
            if isinstance(result, (dict, list)):
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(result)
        except Exception as e:
            print(f"获取资源失败: {str(e)}")
            
    # 调用提示模板
    elif command == "prompt":
        if not args:
            print("错误: 请指定提示模板。")
            print("用法: prompt <服务器名.提示名> [JSON参数]")
            return
            
        # 解析提示规格
        try:
            server_name, prompt_name = parse_server_and_name(args[0])
        except ValueError as e:
            print(f"错误: {str(e)}")
            return
            
        # 解析参数
        params = {}
        if len(args) > 1:
            params_str = " ".join(args[1:])
            try:
                params = parse_json_params(params_str)
            except ValueError as e:
                print(f"错误: {str(e)}")
                return
                
        try:
            result = await client.call_prompt_by_server(server_name, prompt_name, params)
            print("\n提示模板结果:")
            print(result)
        except Exception as e:
            print(f"提示模板调用失败: {str(e)}")
            
    # LLM回答处理
    elif command == "ask":
        if not args:
            print("错误: 请提供要处理的回答。")
            print("用法: ask <回答文本>")
            return
            
        # 提取回答 - 使用原始输入以保留空格和标点
        query = raw_input[len(command):].strip()
        
        try:
            result = await client.smart_query(query)
            print("\n回答结果:")
            print(result)
        except Exception as e:
            print(f"处理回答失败: {str(e)}")
            
    # 清除对话历史
    elif command == "clear-history":
        client.llm_processor.clear_history()
        
    # 未知命令
    else:
        print(f"未知命令: {command}")
        print("输入 'help' 查看可用命令。") 