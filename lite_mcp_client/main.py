"""程序入口点模块"""

import asyncio
import sys
import json
import os
import argparse

from .client import GenericMCPClient
from .config import DEFAULT_CONFIG_PATH


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="通用型 MCP 客户端 - 一个用于连接、调用和管理 MCP 服务的轻量级客户端。",
        epilog="如果不提供参数，或者只提供配置文件路径(-c)，"
               "程序将尝试连接默认服务器并进入交互模式。"
    )
    parser.add_argument(
        '--config', '-c',
        default=DEFAULT_CONFIG_PATH,
        help=f'指定配置文件的路径 (默认: {DEFAULT_CONFIG_PATH})'
    )

    # Connection options
    conn_group = parser.add_argument_group('连接选项')
    conn_group.add_argument(
        '--server',
        metavar='服务器名',
        help='启动时连接到指定的服务器'
    )
    conn_group.add_argument(
        '--connect-all',
        action='store_true',
        help='启动时连接到配置文件中所有标记为默认的服务器'
    )

    # Action options (mutually exclusive group ideally, but complex with interactive fallback)
    action_group = parser.add_argument_group('操作选项 (如果提供，将执行并退出)')
    action_group.add_argument(
        '--query', '--ask', # Allow both --query and --ask
        metavar='自然语言回答',
        help='使用 LLM 处理回答并获取结果'
    )
    action_group.add_argument(
        '--call',
        metavar='服务器名.工具名',
        help='直接调用指定的工具'
    )
    action_group.add_argument(
        '--prompt',
        metavar='服务器名.提示名',
        help='直接调用指定的提示模板'
    )
    action_group.add_argument(
        '--get',
        metavar='服务器名.资源URI',
        help='直接获取指定的资源'
    )
    action_group.add_argument(
        '--params',
        metavar='JSON字符串',
        help='为 --call 或 --prompt 提供参数 (必须是有效的JSON对象)'
    )

    # Mode option
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='强制进入交互模式，即使提供了其他操作参数'
    )

    # Positional argument for simple queries (must be last)
    parser.add_argument(
        'direct_query',
        nargs='?', # Optional positional argument
        metavar='回答文本',
        help='直接提供自然语言回答 (等同于 --query，不需要标志)'
    )

    args = parser.parse_args()

    # --- Client Initialization ---
    client = GenericMCPClient()
    initial_connection_made = False
    exit_code = 0 # Default exit code

    try:
        # --- Load Config ---
        try:
             config = client.load_server_config(args.config)
        except ValueError as e:
             print(f"错误: {e}")
             sys.exit(1) # Exit if config is fundamentally broken

        # --- Initial Connection Logic ---
        if args.server:
            print(f"尝试连接指定服务器: {args.server}")
            initial_connection_made = await client.connect_to_server_by_name(args.server, config)
        elif args.connect_all:
             print("尝试连接所有默认服务器...")
             initial_connection_made = await client.connect_all_default_servers(config)
        elif not any([args.query, args.direct_query, args.call, args.prompt, args.get, args.interactive]):
             # Default behavior: connect to defaults if no action/interactive specified
             print("未指定操作，尝试连接默认服务器...")
             initial_connection_made = await client.connect_all_default_servers(config)


        # --- Action Execution (if not forced interactive) ---
        action_performed = False
        if not args.interactive:
            query_to_run = args.query or args.direct_query
            if query_to_run:
                action_performed = True
                print(f"\n执行LLM回答: {query_to_run}")
                if not client.connections: # Connect if needed
                     print("LLM回答需要连接，尝试连接默认服务器...")
                     await client.connect_all_default_servers(config)
                if not client.connections:
                     print("错误: 无法连接到任何服务器以执行回答。")
                     exit_code = 1
                else:
                     result = await client.smart_query(query_to_run)
                     print("\n回答结果:")
                     print(result)


            elif args.call:
                 action_performed = True
                 if not args.call or '.' not in args.call:
                     print("错误: --call 参数需要格式 '服务器名.工具名'")
                     exit_code = 1
                 else:
                     server_name, tool_name = args.call.split('.', 1)
                     params = {}
                     if args.params:
                         try:
                             params = json.loads(args.params)
                             if not isinstance(params, dict): raise ValueError("参数必须是JSON对象")
                         except (json.JSONDecodeError, ValueError) as e:
                             print(f"错误: 无效的 --params JSON: {e}")
                             exit_code = 1
                     if exit_code == 0:
                          print(f"\n执行工具调用: {server_name}.{tool_name}")
                          if server_name not in client.connections: # Connect if needed
                               print(f"需要连接到 '{server_name}'...")
                               await client.connect_to_server_by_name(server_name, config)
                          try:
                               result = await client.call_tool_by_server(server_name, tool_name, params)
                               print("\n调用结果:")
                               # 直接输出结果，不进行序列化处理
                               print(result)
                          except Exception as e:
                               print(f"\n工具调用失败: {e}")
                               exit_code = 1


            elif args.prompt:
                 action_performed = True
                 if not args.prompt or '.' not in args.prompt:
                     print("错误: --prompt 参数需要格式 '服务器名.提示名'")
                     exit_code = 1
                 else:
                     server_name, prompt_name = args.prompt.split('.', 1)
                     params = {}
                     if args.params:
                          try:
                             params = json.loads(args.params)
                             if not isinstance(params, dict): raise ValueError("参数必须是JSON对象")
                          except (json.JSONDecodeError, ValueError) as e:
                             print(f"错误: 无效的 --params JSON: {e}")
                             exit_code = 1
                     if exit_code == 0:
                          print(f"\n执行提示调用: {server_name}.{prompt_name}")
                          if server_name not in client.connections:
                               print(f"需要连接到 '{server_name}'...")
                               await client.connect_to_server_by_name(server_name, config)
                          try:
                               result = await client.call_prompt_by_server(server_name, prompt_name, params)
                               print("\n调用结果:")
                               print(result)
                          except Exception as e:
                               print(f"\n提示调用失败: {e}")
                               exit_code = 1


            elif args.get:
                 action_performed = True
                 if not args.get or '.' not in args.get:
                     print("错误: --get 参数需要格式 '服务器名.资源URI'")
                     exit_code = 1
                 else:
                     server_name, resource_uri = args.get.split('.', 1)
                     print(f"\n执行资源获取: {server_name}.{resource_uri}")
                     if server_name not in client.connections:
                          print(f"需要连接到 '{server_name}'...")
                          await client.connect_to_server_by_name(server_name, config)
                     try:
                          result = await client.get_resource_by_server(server_name, resource_uri)
                          print("\n获取结果:")
                          if isinstance(result, (dict, list)):
                              print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                          else:
                              print(result)
                     except Exception as e:
                          print(f"\n资源获取失败: {e}")
                          exit_code = 1


        # --- Enter Interactive Mode? ---
        # Enter if:
        # 1. Explicitly requested (--interactive)
        # 2. No action was performed via command line args
        if args.interactive or not action_performed:
            # If no initial connection was made (e.g., only --interactive flag given),
            # maybe enter server management mode first? Or just start interactive.
            if not client.connections:
                print("\n未连接到任何服务器。")
                # Optionally start management mode here:
                # managed_to_connect = await client.server_management_mode(args.config)
                # if not managed_to_connect:
                #     print("未能连接到服务器，退出。")
                # else:
                #     await client.interactive_mode()

                # Simpler: Just start interactive mode, user needs to 'connect'
                await client.interactive_mode()
            else:
                 await client.interactive_mode()


    except KeyboardInterrupt:
         print("\n操作被用户中断。")
         exit_code = 130 # Standard exit code for Ctrl+C
    except Exception as e:
         print(f"\n发生未处理的主程序错误: {str(e)}")
         import traceback
         traceback.print_exc()
         exit_code = 1
    finally:
        # --- Cleanup ---
        print("\n正在关闭客户端并清理连接...")
        try:
            # 较短的超时时间
            await asyncio.wait_for(client.cleanup(), timeout=3.0)
        except asyncio.TimeoutError:
            print("警告: 客户端清理超时")
        except Exception as cleanup_err:
            print(f"清理连接时出错: {cleanup_err}")

        # --- Final Task Cancellation ---
        # 使用更健壮的最终清理方式
        try:
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                for task in tasks:
                    task.cancel()
                # 设置极短的超时，我们只是尝试取消但不等待太久
                await asyncio.wait(tasks, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
        except Exception:
            pass  # 忽略最终清理中的任何错误

        print(f"程序终止。退出代码: {exit_code}")
        # 直接使用os._exit避免进一步的清理逻辑
        os._exit(exit_code)


def entry_point():
    import asyncio
    return asyncio.run(main())


if __name__ == "__main__":
    entry_point() 