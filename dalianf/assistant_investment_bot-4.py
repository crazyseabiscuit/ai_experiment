# -*- coding: utf-8 -*-
"""
多智能体版智能投研助手（分工更清晰）
- 角色分工：需求分析师BA、研究员（只整理资料）、报告员（撰写报告/可扩展图表合规）
- 自动流转：Router自动分发任务
- 工具注册：planner、tavily-mcp、data_organizer、reporter
- 支持WebUI交互

Version4说明：
多智能体分工更清晰：同样有BA、研究员、报告员三个Agent，但每个角色的职责更加明确。
流程强制分段：
BA：只输出报告框架和调研清单，明确提示下一步应由研究员搜集资料，不直接让报告员写报告。
研究员：只负责资料搜集和整理（用 data_organizer 工具），不写完整报告。资料整理为“调研问题-答案”结构。
报告员：只根据BA框架和研究员整理的资料撰写最终报告，并可扩展为图表、合规检查等。
循环与调用次数限制：研究员的 system prompt 明确限制 tavily-mcp 工具调用次数（如不超过10次），最多2轮检索与整理，防止死循环。
"""
import os
from qwen_agent.agents import Assistant, Router
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ========== 工具注册 ==========
# planner 工具
@register_tool('planner')
class PlannerTool(BaseTool):
    """
    投研规划工具，调用大模型（qwen-turbo），根据 planner.md 的提示词撰写投研报告框架。
    """
    description = '投研规划工具，输出投研框架。'
    parameters = [
        {
            'name': 'user_input',
            'type': 'string',
            'description': '用户的投研需求描述',
            'required': True
        }
    ]

    @staticmethod
    def get_prompt():
        with open('planner.md', 'r', encoding='utf-8') as f:
            prompt = f.read()
        today = datetime.now().strftime("%Y年%m月%d日")
        return prompt.format(CURRENT_TIME=today)

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        user_input = args['user_input']
        prompt = self.get_prompt()
        import dashscope
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=messages,
            result_format='message'
        )
        return response.output.choices[0].message.content

# reporter 工具
@register_tool('reporter')
class ReporterTool(BaseTool):
    """
    投研报告撰写工具，根据 planner 整理并用户确认过的框架，生成最终投研报告。
    """
    description = '投研报告撰写工具，严格按照已确认的框架生成报告。'
    parameters = [
        {
            'name': 'framework',
            'type': 'string',
            'description': '已确认的投研报告框架（由planner输出并经用户确认）',
            'required': True
        },
        {
            'name': 'research_data',
            'type': 'string',
            'description': '调研和信息收集的主要内容（可选，若有信息收集结果可传入）',
            'required': False
        }
    ]

    @staticmethod
    def get_prompt():
        with open('reporter.md', 'r', encoding='utf-8') as f:
            prompt = f.read()
        today = datetime.now().strftime("%Y年%m月%d日")
        return prompt.format(CURRENT_TIME=today)

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        framework = args['framework']
        research_data = args.get('research_data', '')
        prompt = self.get_prompt()
        import dashscope
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"报告框架如下：\n{framework}\n\n调研内容如下：\n{research_data}"}
        ]
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=messages,
            result_format='message'
        )
        return response.output.choices[0].message.content

# 资料整理工具
@register_tool('data_organizer')
class DataOrganizerTool(BaseTool):
    """
    资料整理工具：根据调研清单和原始资料，整理出结构化的调研问题-答案对。
    """
    description = '资料整理工具，根据调研清单和原始资料，输出调研问题-答案结构。'
    parameters = [
        {
            'name': 'research_questions',
            'type': 'string',
            'description': '调研清单（问题列表）',
            'required': True
        },
        {
            'name': 'raw_materials',
            'type': 'string',
            'description': 'tavily-mcp等工具搜集的原始资料',
            'required': True
        }
    ]

    @staticmethod
    def get_prompt():
        return (
            "你是一名资料整理员，请根据调研清单和原始资料，整理出结构化的调研问题-答案对。\n"
            "每个问题对应一段简明、客观的资料总结，不要撰写完整报告。\n"
            "输出格式：\n"
            "# 调研问题-答案整理\n"
            "1. 问题：xxx\n   回答：xxx\n2. 问题：yyy\n   回答：yyy\n..."
        )

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        research_questions = args['research_questions']
        raw_materials = args['raw_materials']
        prompt = self.get_prompt()
        import dashscope
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"调研清单：\n{research_questions}\n\n原始资料：\n{raw_materials}"}
        ]
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=messages,
            result_format='message'
        )
        return response.output.choices[0].message.content

# ========== tavily-mcp 工具外部服务注册 ==========
tools = [{
    "mcpServers": {
        "tavily-mcp": {
            "command": "npx",
            "args": ["-y", "tavily-mcp@0.2.0"],
            "env": {
                "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "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")
            }
        }
    }
}]

# ========== system prompt 定义 ==========
BA_PROMPT = (
    "你是需求分析师BA，负责撰写投研报告框架和调研清单，并与用户确认。"
    "请只输出报告框架和调研清单，下一步应由研究员根据调研清单搜集资料并整理，不要直接让报告员写报告。"
)
RESEARCHER_PROMPT = (
    "你是研究员，只负责根据BA的调研清单，调用tavily-mcp等工具搜集资料，并用资料整理工具将原始资料整理为调研问题-答案对。"
    "不要撰写完整报告，只输出结构化的调研问题-答案整理。"
    "如有部分问题无法整理出答案，最多尝试2轮检索与整理，若仍无结果请如实告知'资料暂未覆盖该问题'。"
    "tavily-mcp工具总调用次数不得超过10次。"
    "当你完成后，CALL 报告员"
)
REPORTER_PROMPT = (
    "你是报告员，首先根据BA的报告框架和研究员整理的调研问题-答案，撰写完整的投研报告。"
    "如有需要，后续可进一步生成图表、合规检查等专业内容，但请务必先完成报告正文。"
)

# ========== Agent 初始化 ==========
def init_agents():
    ba_agent = Assistant(
        llm={'model': 'qwen-turbo-latest', 'timeout': 30},
        name='需求分析师BA',
        description='撰写投研报告框架和调研清单',
        system_message=BA_PROMPT,
        function_list=['planner']
    )
    researcher_agent = Assistant(
        llm={'model': 'qwen-turbo-latest', 'timeout': 30},
        name='研究员',
        description='只整理资料，不写报告',
        system_message=RESEARCHER_PROMPT,
        function_list=tools + ['data_organizer']
    )
    reporter_agent = Assistant(
        llm={'model': 'qwen-turbo-latest', 'timeout': 30},
        name='报告员',
        description='撰写最终投研报告',
        system_message=REPORTER_PROMPT,
        function_list=['reporter']
    )
    return ba_agent, researcher_agent, reporter_agent

# ========== Router 初始化 ==========
def init_router():
    ba_agent, researcher_agent, reporter_agent = init_agents()
    router = Router(
        llm={'model': 'qwen-turbo-latest', 'timeout': 30},
        agents=[ba_agent, researcher_agent, reporter_agent],
        name='投研助手Router',
        description='自动流转BA-研究员-报告员流程'
    )
    return router

# ========== WebUI 启动 ==========
def app_gui():
    bot = init_router()
    chatbot_config = {
        'prompt.suggestions': [
            '撰写 中芯国际 投研报告',
            '请根据最新的行业报告，分析国产芯片的市场空间和主要驱动力',
            '请结合财报和券商研报，写一份中芯国际的投资价值分析',
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run(server_name="0.0.0.0", server_port=9002, share=False)    

if __name__ == '__main__':
    app_gui() 