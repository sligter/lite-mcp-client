"""LLM处理和智能回答模块"""

import json
import re
from typing import Dict, Any, List, Tuple, Optional

class LLMProcessor:
    """处理LLM回答和工具调用"""
    
    def __init__(self, client):
        """初始化LLM处理器
        
        Args:
            client: GenericMCPClient的实例，用于访问连接、工具等
        """
        self.client = client
        self.history = []  # 会话历史
    
    def clear_history(self):
        """清除对话历史"""
        self.history = []
        print("对话历史已清除。")
    
    async def smart_query(self, query_text: str, max_attempts: int = 30) -> str:
        """LLM回答处理器 - 调用LLM解释回答并调用适当的工具
        
        Args:
            query_text: 用户的回答
            max_attempts: 最大工具调用尝试次数
            
        Returns:
            最终回答
        """
        if not self.client.connections:
            print("提示: 没有已连接的服务器，使用纯模型回答")

        print(f"\n正在处理LLM回答: {query_text}\n")

        # 构建上下文
        available_tools = self.client.collect_all_tools()
        available_prompts = self.client.collect_all_prompts()
        available_resources = self.client.collect_all_resources()

        # 添加用户回答到历史记录
        self.history.append({"role": "user", "content": query_text})
        
        # 尝试循环
        attempt = 0
        final_response = None
        
        # 在调用LLM前添加调试信息
        print(f"可用工具数量: {sum(len(tools) for tools in available_tools.values())}")
        print(f"可用提示模板数量: {sum(len(prompts) for prompts in available_prompts.values())}")
        
        while attempt < max_attempts:
            attempt += 1
            
            if attempt > 1:
                print(f"\n尝试 {attempt}/{max_attempts}")
            
            # 调用LLM获取下一步操作
            next_step = await self._call_llm_for_next_step(
                self.history, available_tools, available_prompts, available_resources
            )

            # 打印原始LLM响应（调试用）
            print(f"\nLLM原始响应: {next_step.get('raw_response', '')}")
            
            action = next_step.get("action", "")
            
            if action == "final_answer":
                final_response = next_step.get("content", "")
                # 添加到历史
                self.history.append({"role": "assistant", "content": final_response})
                break
                
            elif action == "call_tool":
                server_name = next_step.get("server", "")
                tool_name = next_step.get("tool", "")
                params = next_step.get("params", {})
                
                print(f"\n正在调用工具: {server_name}.{tool_name}")
                print(f"参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
                
                try:
                    # 实际调用工具
                    result = await self.client.call_tool_by_server(server_name, tool_name, params)
                    
                    # 确保结果是可序列化的类型
                    json_safe_result = self._ensure_json_serializable(result)
                    
                    print(f"\n工具调用结果:\n{json.dumps(json_safe_result, ensure_ascii=False, indent=2)}")
                    
                    # 添加结果到历史
                    tool_message = (
                        f"工具 {server_name}.{tool_name} 的调用结果:\n\n"
                        f"```\n{json.dumps(json_safe_result, ensure_ascii=False, indent=2)}\n```"
                    )
                    self.history.append({"role": "user", "content": tool_message})
                    
                except Exception as e:
                    error_msg = f"工具调用失败: {str(e)}"
                    print(f"\n{error_msg}")
                    self.history.append({"role": "user", "content": ""})
                    
            elif action == "call_prompt":
                server_name = next_step.get("server", "")
                prompt_name = next_step.get("prompt", "")
                params = next_step.get("params", {})
                
                print(f"\n正在调用提示模板: {server_name}.{prompt_name}")
                print(f"参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
                
                try:
                    # 实际调用提示模板
                    result = await self.client.call_prompt_by_server(server_name, prompt_name, params)
                    
                    print(f"\n提示模板结果:\n{result}")
                    
                    # 添加结果到历史
                    prompt_message = (
                        f"使用提示模板 {server_name}.{prompt_name} 的结果:\n\n"
                        f"```\n{result}\n```"
                    )
                    self.history.append({"role": "user", "content": prompt_message})
                    
                except Exception as e:
                    error_msg = f"提示模板调用失败: {str(e)}"
                    print(f"\n{error_msg}")
                    self.history.append({"role": "user", "content": ""})
                    
            elif action == "error":
                error_msg = next_step.get("error", "未知错误")
                print(f"\n处理错误: {error_msg}")
                
                # 如果是重大错误，只尝试一次重试
                if attempt > 2:
                    final_response = f"抱歉，在处理您的回答时遇到了问题: {error_msg}"
                    # 添加到历史
                    self.history.append({"role": "assistant", "content": final_response})
                    break
                    
                # 添加错误信息到历史
                self.history.append({"role": "user", "content": f"出现错误: {error_msg}。请重新尝试。"})
            else:
                print(f"\n未知操作类型: {action}")
                final_response = "抱歉，处理您的回答时出现了内部错误。"
                # 添加到历史
                self.history.append({"role": "assistant", "content": final_response})
                break
                
        # 如果达到最大尝试次数仍未得到答案
        if attempt >= max_attempts and not final_response:
            final_response = "抱歉，似乎我无法完成这个回答处理。请尝试简化您的回答或者以不同方式提问。"
            # 添加到历史
            self.history.append({"role": "assistant", "content": final_response})
            
        return final_response or "抱歉，无法生成回答。"

    def _format_tools_for_llm(self) -> str:
        """格式化工具信息，方便LLM使用
        
        Returns:
            格式化的工具信息字符串
        """
        all_tools = self.client.collect_all_tools()
        if not all_tools:
            return "没有可用工具。"
            
        tools_text = []
        
        for server_name, tools in all_tools.items():
            for tool in tools:
                tool_name = tool.get("name", "未知工具")
                full_name = f"{server_name}.{tool_name}"
                description = tool.get("description", "无描述")
                
                tool_text = [f"- {full_name}: {description}"]
                
                # 添加参数信息
                schema = tool.get("schema", {})
                if isinstance(schema, dict) and "properties" in schema:
                    properties = schema.get("properties", {})
                    required = schema.get("required", [])
                    
                    if properties:
                        tool_text.append("  参数:")
                        for param_name, param_info in properties.items():
                            param_type = param_info.get("type", "any")
                            param_desc = param_info.get("description", "")
                            req_marker = "*" if param_name in required else ""
                            
                            tool_text.append(f"    - {param_name}{req_marker} ({param_type}): {param_desc}")
                            
                tools_text.append("\n".join(tool_text))
                
        return "\n\n".join(tools_text)
    
    def _format_prompts_for_llm(self) -> str:
        """格式化提示模板信息，方便LLM使用
        
        Returns:
            格式化的提示模板信息字符串
        """
        all_prompts = self.client.collect_all_prompts()
        if not all_prompts:
            return "没有可用提示模板。"
            
        prompts_text = []
        
        for server_name, prompts in all_prompts.items():
            for prompt in prompts:
                prompt_name = prompt.get("name", "未知提示模板")
                full_name = f"{server_name}.{prompt_name}"
                description = prompt.get("description", "无描述")
                
                prompt_text = [f"- {full_name}: {description}"]
                
                # 添加参数信息
                schema = prompt.get("schema", {})
                if isinstance(schema, dict) and "properties" in schema:
                    properties = schema.get("properties", {})
                    required = schema.get("required", [])
                    
                    if properties:
                        prompt_text.append("  参数:")
                        for param_name, param_info in properties.items():
                            param_type = param_info.get("type", "any")
                            param_desc = param_info.get("description", "")
                            req_marker = "*" if param_name in required else ""
                            
                            prompt_text.append(f"    - {param_name}{req_marker} ({param_type}): {param_desc}")
                            
                prompts_text.append("\n".join(prompt_text))
                
        return "\n\n".join(prompts_text)

    async def _call_llm_for_next_step(self, transcript, available_tools, available_prompts, available_resources):
        """调用 LLM 确定下一步行动"""
        if not self.client.llm:
            return {"action": "error", "error": "错误: LLM 未初始化，无法执行LLM回答。"}
        
        # 格式化工具和提示信息
        tools_info = self._format_tools_for_llm()
        prompts_info = self._format_prompts_for_llm()
        
        # 准备系统提示 - 注意所有JSON示例中的大括号要用双大括号转义
        system_prompt = f"""你是一个全能助手，可以在有需要时使用以下工具来回答用户问题。

--- 可用工具 ---
{tools_info}
--- 可用提示模板 ---
{prompts_info}
---

你的思考和行动步骤:
1.  **分析请求:** 理解用户的问题。
2.  **规划:** 决定是否需要调用工具或提示模板。如果需要，选择最合适的，并确定所需的参数。参数名旁边带 * 号的是必需的。
3.  **行动:**
    *   **调用工具:** 如果需要调用工具，使用以下精确格式（包括代码块标记）。一次只能调用一个工具。
        ```json
        ACTION: call_tool
        {{
          "tool": "服务器名.工具名",
          "params": {{
            "参数名1": "参数值1",
            "参数名2": "参数值2"
          }}
        }}
        ```
    *   **调用提示:** 如果需要使用提示模板，使用以下精确格式。
        ```json
        ACTION: call_prompt
        {{
          "prompt": "服务器名.提示名",
          "params": {{
            "参数名1": "参数值1"
          }}
        }}
        ```
    *   **回答:** 当你收集到足够的信息或者认为不需要调用工具时，直接回答用户。使用以下格式：
        ```text
        ACTION: final_answer
        <这里是给用户的最终回答>
        ```
4.  **迭代:** 在收到工具或提示的调用结果后，重复步骤 1-3，直到任务完成。

**重要规则:**
*   **必需参数:** 确保为标记为 * 的必需参数提供值。
*   **一次一个 Action:** 每次回复只能包含一个 `ACTION`。
*   **精确格式:** 严格遵守 `ACTION` 的 JSON 格式。JSON 必须有效。
*   **服务器名:** 必须在 `tool` 或 `prompt` 字段中包含服务器名 (例如 `服务器A.get_news`)。
*   **等待结果:** 在每次调用工具或提示后，我会提供结果，请等待结果再进行下一步。
*   **最终答案:** 只有在你完成所有必要步骤并准备好最终答案时，才使用 `ACTION: final_answer`。
"""
        
        # 准备消息列表
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 添加对话历史
        for msg in transcript:
            messages.append(msg)
            
        # 调用LLM
        llm_response = None
        try:
            # 转换消息格式以适应不同的LLM接口
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    from langchain_core.messages import SystemMessage
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    from langchain_core.messages import HumanMessage
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    from langchain_core.messages import AIMessage
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            response = await self.client.llm.ainvoke(langchain_messages)
            llm_content = response.content.strip()
            
            # 保存原始响应
            llm_response = llm_content
            
            # 解析响应，查找ACTION标记
            action_match = re.search(r'ACTION:\s*(.*?)(?:\n|$)', llm_content, re.IGNORECASE)
            if not action_match:
                print("未找到ACTION标记，默认为最终答案")
                return {
                    "action": "final_answer",
                    "content": llm_content,
                    "raw_response": llm_content
                }
            
            action_type = action_match.group(1).strip().lower()
            
            if "final_answer" in action_type:
                # 提取最终答案内容
                final_answer_match = re.search(r'ACTION:\s*final_answer\s*(.*)', llm_content, re.DOTALL | re.IGNORECASE)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                else:
                    final_answer = llm_content.replace("ACTION: final_answer", "").strip()
                
                return {
                    "action": "final_answer",
                    "content": final_answer,
                    "raw_response": llm_content
                }
            
            elif "call_tool" in action_type:
                # 从响应中提取JSON
                json_text, json_data = self._extract_json_from_text(llm_content)
                
                if not json_data:
                    json_data = self._emergency_parse_json(llm_content)
                    
                    if not json_data:
                        return {
                            "action": "error",
                            "error": "无法从LLM响应中提取有效的JSON数据",
                            "raw_response": llm_content
                        }

                tool_spec = json_data.get("tool")
                params = json_data.get("params", {})
                
                if not tool_spec:
                    return {
                        "action": "error",
                        "error": "工具调用缺少工具名称",
                        "raw_response": llm_content
                    }
                
                # 解析服务器名和工具名
                if '.' in tool_spec:
                    server_name, tool_name = tool_spec.split('.', 1)
                else:
                    if not self.client.connections:
                        return {
                            "action": "error",
                            "error": "未指定服务器名，且没有连接到任何服务器",
                            "raw_response": llm_content
                        }
                    server_name = next(iter(self.client.connections.keys()))
                    tool_name = tool_spec
                
                return {
                    "action": "call_tool",
                    "server": server_name,
                    "tool": tool_name,
                    "params": params,
                    "raw_response": llm_content
                }
            
            elif "call_prompt" in action_type:
                # 从响应中提取JSON
                json_text, json_data = self._extract_json_from_text(llm_content)

                if not json_data:
                    json_data = self._emergency_parse_json(llm_content)
                    
                    if not json_data:
                        return {
                            "action": "error",
                            "error": "无法从LLM响应中提取有效的JSON数据",
                            "raw_response": llm_content
                        }

                prompt_spec = json_data.get("prompt")
                params = json_data.get("params", {})
                
                if not prompt_spec:
                    return {
                        "action": "error",
                        "error": "提示调用缺少提示模板名称",
                        "raw_response": llm_content
                    }
                
                # 解析服务器名和提示名
                if '.' in prompt_spec:
                    server_name, prompt_name = prompt_spec.split('.', 1)
                else:
                    if not self.client.connections:
                        return {
                            "action": "error",
                            "error": "未指定服务器名，且没有连接到任何服务器",
                            "raw_response": llm_content
                        }
                    server_name = next(iter(self.client.connections.keys()))
                    prompt_name = prompt_spec
                
                return {
                    "action": "call_prompt",
                    "server": server_name,
                    "prompt": prompt_name,
                    "params": params,
                    "raw_response": llm_content
                }
            
            else:
                # 未知的动作类型
                return {
                    "action": "error",
                    "error": f"未知的动作类型: {action_type}",
                    "raw_response": llm_content
                }
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "action": "error",
                "error": f"处理LLM响应时出错: {str(e)}",
                "raw_response": llm_response or "LLM调用失败"
            }

    def _extract_json_from_text(self, text: str) -> Tuple[str, Dict]:
        """从文本中提取JSON内容"""
        # 在处理前先检查和清理输入文本
        if not text or not isinstance(text, str):
            return "", {}
        
        # 如果文本包含多余的反斜杠，尝试消除它们
        if '\\' in text and '\\\\' not in text:
            cleaned_text = text.replace('\\"', '"')
            try:
                parsed_json = json.loads(cleaned_text)
                return cleaned_text, parsed_json
            except json.JSONDecodeError:
                pass  # 继续尝试其他策略
        
        # 尝试不同的提取策略
        strategies = [
            # 查找代码块中的JSON
            lambda t: re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', t),
            # 查找{}包围的内容
            lambda t: re.search(r'({[\s\S]*?})', t),
            # 查找第一个{到最后一个}
            lambda t: re.search(r'({[\s\S]*})', t) if '{' in t and '}' in t else None
        ]
        
        json_text = ""
        parsed_json = {}
        
        for strategy in strategies:
            match = strategy(text)
            if match:
                json_text = match.group(1).strip()
                try:
                    parsed_json = json.loads(json_text)
                    return json_text, parsed_json
                except json.JSONDecodeError:
                    # 尝试清理JSON文本
                    cleaned_json = self._clean_json_text(json_text)
                    try:
                        parsed_json = json.loads(cleaned_json)
                        return cleaned_json, parsed_json
                    except json.JSONDecodeError:
                        # 继续尝试下一个策略
                        continue
        
        # 所有策略都失败，尝试整个文本
        try:
            parsed_json = json.loads(text)
            return text, parsed_json
        except json.JSONDecodeError:
            # 尝试清理整个文本
            cleaned_text = self._clean_json_text(text)
            try:
                parsed_json = json.loads(cleaned_text)
                return cleaned_text, parsed_json
            except json.JSONDecodeError:
                # 所有尝试都失败
                return json_text, {}

    def _clean_json_text(self, text: str) -> str:
        """清理JSON文本，移除可能导致解析错误的内容
        
        Args:
            text: 需要清理的JSON文本
            
        Returns:
            清理后的JSON文本
        """
        # 移除可能的Markdown格式
        text = re.sub(r'```.*\n|```', '', text)
        
        # 移除多余的空格和换行
        text = text.strip()
        
        # 替换不规范的JSON语法
        text = re.sub(r'(\w+):', r'"\1":', text)  # 将 key: 替换为 "key":
        text = re.sub(r',\s*}', '}', text)        # 移除对象末尾的逗号
        text = re.sub(r',\s*]', ']', text)        # 移除数组末尾的逗号
        
        # 处理可能的多行字符串格式问题
        text = re.sub(r'"""(.*?)"""', r'"\1"', text, flags=re.DOTALL)
        text = re.sub(r"'''(.*?)'''", r'"\1"', text, flags=re.DOTALL)
        
        return text

    def _emergency_parse_json(self, text: str) -> Dict:
        """当正常JSON解析失败时的紧急解析方法
        
        Args:
            text: 可能包含JSON的文本
            
        Returns:
            解析的JSON字典或空字典
        """
        try:
            # 1. 尝试查找所有 { } 块
            brace_pattern = re.compile(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}')
            matches = brace_pattern.findall(text)
            
            for match in matches:
                try:
                    # 尝试直接解析
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        return parsed
                except:
                    # 尝试清理后解析
                    cleaned = re.sub(r'(\w+):', r'"\1":', match)  # key -> "key"
                    cleaned = re.sub(r',\s*}', '}', cleaned)      # remove trailing commas
                    try:
                        parsed = json.loads(cleaned)
                        if isinstance(parsed, dict) and len(parsed) > 0:
                            return parsed
                    except:
                        continue  # 尝试下一个匹配
            
            # 2. 如果没有找到有效的JSON，尝试截取 "tool" 或 "prompt" 相关内容
            tool_pattern = re.compile(r'"tool"\s*:\s*"([^"]+)"')
            tool_match = tool_pattern.search(text)
            
            prompt_pattern = re.compile(r'"prompt"\s*:\s*"([^"]+)"')
            prompt_match = prompt_pattern.search(text)
            
            params_pattern = re.compile(r'"params"\s*:\s*({[^{}]*(?:{[^{}]*}[^{}]*)*})')
            params_match = params_pattern.search(text)
            
            result = {}
            
            if tool_match:
                result["tool"] = tool_match.group(1)
            elif prompt_match:
                result["prompt"] = prompt_match.group(1)
                
            if params_match:
                try:
                    params_text = params_match.group(1)
                    # 尝试清理并解析参数
                    cleaned_params = re.sub(r'(\w+):', r'"\1":', params_text)
                    cleaned_params = re.sub(r',\s*}', '}', cleaned_params)
                    params = json.loads(cleaned_params)
                    result["params"] = params
                except:
                    result["params"] = {}
            
            return result
        except Exception as e:
            print(f"紧急JSON解析也失败: {e}")
            return {}

    def _ensure_json_serializable(self, obj):
        """确保对象是JSON可序列化的
        
        Args:
            obj: 任何类型的对象
            
        Returns:
            JSON可序列化的对象
        """
        if obj is None:
            return None
        
        # 处理常见的简单类型
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        
        # 处理列表
        if isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        
        # 处理字典
        if isinstance(obj, dict):
            return {str(k): self._ensure_json_serializable(v) for k, v in obj.items()}
        
        # 处理TextContent或其他MCP对象
        if hasattr(obj, 'content'):
            return obj.content
        
        # 处理有to_dict方法的对象
        if hasattr(obj, 'to_dict') and callable(obj.to_dict):
            return obj.to_dict()
        
        # 其他对象尝试转为字符串
        try:
            return str(obj)
        except:
            return f"<不可序列化对象: {type(obj).__name__}>" 