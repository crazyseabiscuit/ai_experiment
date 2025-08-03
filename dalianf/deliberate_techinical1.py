#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术调研智能体（Tech Research Agent）- Agent比赛技术解决方案调研

基于LangGraph实现的深思熟虑型技术调研智能体，专门用于调研最新Agent比赛的
获奖技术解决方案，分析技术趋势和创新点。

重构版本特点：
- 模块化设计，职责分离
- 配置化管理，易于扩展
- 错误处理完善
- 类型提示完整
- 文档注释详细
"""

import os
import json
import logging
from typing import Dict, List, Any, Literal, TypedDict, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langgraph.graph import StateGraph, END

# ========== 配置管理 ==========
@dataclass
class AgentConfig:
    """智能体配置类"""
    model_name: str = "Qwen-Turbo-2025-04-28"
    timeout: int = 30
    max_retries: int = 3
    api_key_env: str = "DASHSCOPE_API_KEY"
    
    # 调研主题预设
    preset_topics: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        if self.preset_topics is None:
            self.preset_topics = {
                "1": {
                    "topic": "AgentBench 2024 获奖解决方案技术分析",
                    "competition": "AgentBench 2024",
                    "period": "2024年"
                },
                "2": {
                    "topic": "AutoGen 多智能体协作技术调研",
                    "competition": "AutoGen相关比赛",
                    "period": "2023-2024年"
                },
                "3": {
                    "topic": "LangGraph 在Agent比赛中的应用",
                    "competition": "LangGraph相关比赛",
                    "period": "2024年"
                }
            }

# ========== 枚举定义 ==========
class ResearchPhase(Enum):
    """调研阶段枚举"""
    PERCEPTION = "perception"
    MODELING = "modeling"
    REASONING = "reasoning"
    DECISION = "decision"
    REPORT = "report"
    COMPLETED = "completed"

# ========== 数据模型 ==========
class CompetitionInfo(BaseModel):
    """比赛信息收集"""
    competition_name: str = Field(..., description="比赛名称")
    competition_scope: str = Field(..., description="比赛范围和主题")
    winning_teams: List[Dict[str, str]] = Field(..., description="获奖团队信息")
    key_metrics: Dict[str, str] = Field(..., description="关键评估指标")
    competition_trends: List[str] = Field(..., description="比赛技术趋势")

class TechnicalSolution(BaseModel):
    """技术解决方案分析"""
    solution_name: str = Field(..., description="解决方案名称")
    team_name: str = Field(..., description="团队名称")
    core_technology: str = Field(..., description="核心技术")
    architecture: str = Field(..., description="技术架构")
    innovation_points: List[str] = Field(..., description="创新点")
    performance_metrics: Dict[str, str] = Field(..., description="性能指标")
    code_repository: Optional[str] = Field(None, description="代码仓库链接")

class TechnologyAnalysis(BaseModel):
    """技术分析模型"""
    dominant_frameworks: List[str] = Field(..., description="主流框架")
    emerging_technologies: List[str] = Field(..., description="新兴技术")
    architecture_patterns: List[str] = Field(..., description="架构模式")
    evaluation_methods: List[str] = Field(..., description="评估方法")
    implementation_challenges: List[str] = Field(..., description="实现挑战")

class SolutionComparison(BaseModel):
    """解决方案对比分析"""
    comparison_id: str = Field(..., description="对比ID")
    solution_a: str = Field(..., description="方案A")
    solution_b: str = Field(..., description="方案B")
    comparison_dimensions: List[str] = Field(..., description="对比维度")
    advantages_a: List[str] = Field(..., description="方案A优势")
    advantages_b: List[str] = Field(..., description="方案B优势")
    recommendation: str = Field(..., description="推荐建议")

class ResearchRecommendation(BaseModel):
    """调研推荐"""
    top_solutions: List[str] = Field(..., description="推荐解决方案")
    learning_path: List[str] = Field(..., description="学习路径")
    implementation_priority: List[str] = Field(..., description="实施优先级")
    risk_assessment: str = Field(..., description="风险评估")
    resource_requirements: Dict[str, str] = Field(..., description="资源需求")

# 定义智能体状态
class TechResearchAgentState(TypedDict):
    """技术调研智能体的状态"""
    # 输入
    research_topic: str  # 调研主题
    competition_focus: str  # 比赛焦点
    time_period: str  # 时间范围
    
    # 处理状态
    competition_data: Optional[Dict[str, Any]]  # 比赛信息数据
    technical_model: Optional[Dict[str, Any]]  # 技术架构模型
    solution_analysis: Optional[List[Dict[str, Any]]]  # 解决方案分析
    comparison_results: Optional[List[Dict[str, Any]]]  # 对比结果
    selected_solutions: Optional[Dict[str, Any]]  # 选中的解决方案
    
    # 输出
    final_report: Optional[str]  # 最终调研报告
    
    # 控制流
    current_phase: Literal["perception", "modeling", "reasoning", "decision", "report"]
    error: Optional[str]  # 错误信息

# ========== 提示模板管理 ==========
class PromptTemplates:
    """提示模板管理类"""
    
    @staticmethod
    def get_perception_prompt() -> str:
        """获取感知阶段提示模板"""
        return """你是一个专业的技术调研分析师，请收集和整理关于以下Agent比赛的技术信息：

调研主题: {research_topic}
比赛焦点: {competition_focus}
时间范围: {time_period}

请从以下几个方面收集信息：
1. 比赛基本信息（名称、范围、主题）
2. 获奖团队和解决方案
3. 关键评估指标和评分标准
4. 技术趋势和创新点
5. 相关论文和技术文档

重点关注：
- 最新的Agent比赛（如AgentBench、AgentCoder、AutoGen等）
- 获奖解决方案的技术架构
- 开源代码和实现细节
- 性能评估方法

输出格式要求为JSON，包含以下字段：
- competition_name: 字符串
- competition_scope: 字符串
- winning_teams: 字典列表，包含团队名称、排名、解决方案名称
- key_metrics: 字典，键为指标名称，值为指标说明
- competition_trends: 字符串列表
"""

    @staticmethod
    def get_modeling_prompt() -> str:
        """获取建模阶段提示模板"""
        return """你是一个资深技术架构师，请根据以下比赛信息，构建技术架构模型：

调研主题: {research_topic}
比赛焦点: {competition_focus}
时间范围: {time_period}

比赛信息:
{competition_data}

请构建一个全面的技术架构模型，包括：
1. 主流技术框架和工具
2. 新兴技术和创新点
3. 常见的架构模式
4. 评估方法和指标
5. 实现挑战和解决方案

重点关注：
- LangGraph、AutoGen、CrewAI等框架
- 多智能体协作模式
- 工具使用和集成
- 推理和决策机制
- 性能优化技术

输出格式要求为JSON，包含以下字段：
- dominant_frameworks: 字符串列表
- emerging_technologies: 字符串列表
- architecture_patterns: 字符串列表
- evaluation_methods: 字符串列表
- implementation_challenges: 字符串列表
"""

    @staticmethod
    def get_reasoning_prompt() -> str:
        """获取推理阶段提示模板"""
        return """你是一个技术解决方案分析师，请根据以下技术模型，分析具体的获奖解决方案：

调研主题: {research_topic}
比赛焦点: {competition_focus}
时间范围: {time_period}

技术架构模型:
{technical_model}

请选择3-5个代表性的获奖解决方案进行深入分析，为每个方案提供：
1. 解决方案名称和团队
2. 核心技术栈
3. 技术架构设计
4. 主要创新点
5. 性能指标
6. 代码仓库链接（如果有）

重点关注：
- 技术选型的合理性
- 架构设计的创新性
- 性能优化的方法
- 可扩展性和可维护性
- 开源程度和社区影响

输出格式要求为JSON数组，每个元素包含以下字段：
- solution_name: 字符串
- team_name: 字符串
- core_technology: 字符串
- architecture: 字符串
- innovation_points: 字符串列表
- performance_metrics: 字典
- code_repository: 字符串（可选）
"""

    @staticmethod
    def get_decision_prompt() -> str:
        """获取决策阶段提示模板"""
        return """你是一个技术决策顾问，请评估以下技术解决方案，选择最有价值的技术方案：

调研主题: {research_topic}
比赛焦点: {competition_focus}
时间范围: {time_period}

技术架构模型:
{technical_model}

解决方案分析:
{solution_analysis}

请基于以下维度进行对比分析：
1. 技术创新性
2. 实现复杂度
3. 性能表现
4. 可扩展性
5. 社区支持
6. 学习成本
7. 适用场景

为每对解决方案提供详细的对比分析，并给出推荐建议。

输出格式要求为JSON数组，每个元素包含以下字段：
- comparison_id: 字符串
- solution_a: 字符串
- solution_b: 字符串
- comparison_dimensions: 字符串列表
- advantages_a: 字符串列表
- advantages_b: 字符串列表
- recommendation: 字符串
"""

    @staticmethod
    def get_report_prompt() -> str:
        """获取报告阶段提示模板"""
        return """你是一个技术调研报告撰写专家，请根据以下信息生成一份完整的技术调研报告：

调研主题: {research_topic}
比赛焦点: {competition_focus}
时间范围: {time_period}

比赛信息:
{competition_data}

技术架构模型:
{technical_model}

解决方案分析:
{solution_analysis}

对比分析结果:
{comparison_results}

请生成一份结构完整、内容详实的技术调研报告，包括但不限于：
1. 执行摘要
2. 调研背景和目标
3. 比赛概况和趋势分析
4. 技术架构分析
5. 获奖解决方案深度分析
6. 技术对比和评估
7. 推荐方案和实施建议
8. 风险评估和资源需求
9. 结论和展望

报告应当：
- 技术深度和专业性
- 数据支撑和案例丰富
- 实用性强，可指导实践
- 结构清晰，逻辑严密
"""

# ========== LLM管理 ==========
class LLMManager:
    """LLM管理器"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = self._create_llm()
    
    def _create_llm(self) -> Tongyi:
        """创建LLM实例"""
        api_key = os.getenv(self.config.api_key_env, '')
        if not api_key:
            raise ValueError(f"未设置环境变量 {self.config.api_key_env}")
        
        return Tongyi(
            model_name=self.config.model_name,
            dashscope_api_key=api_key,
            timeout=self.config.timeout
        )
    
    def get_llm(self) -> Tongyi:
        """获取LLM实例"""
        return self.llm

# ========== 阶段处理器 ==========
class PhaseProcessor:
    """阶段处理器基类"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.llm = llm_manager.get_llm()
    
    def process(self, state: TechResearchAgentState) -> TechResearchAgentState:
        """处理阶段逻辑"""
        raise NotImplementedError

class PerceptionProcessor(PhaseProcessor):
    """感知阶段处理器"""
    
    def process(self, state: TechResearchAgentState) -> TechResearchAgentState:
        """感知阶段：收集和整理比赛信息"""
        print("1. 感知阶段：收集Agent比赛信息...")
        
        try:
            # 准备提示
            prompt = ChatPromptTemplate.from_template(PromptTemplates.get_perception_prompt())
            
            # 构建输入
            input_data = {
                "research_topic": state["research_topic"],
                "competition_focus": state["competition_focus"],
                "time_period": state["time_period"]
            }
            
            # 调用LLM
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke(input_data)
            
            # 更新状态
            return {
                **state,
                "competition_data": result,
                "current_phase": ResearchPhase.MODELING.value
            }
        except Exception as e:
            logging.error(f"感知阶段出错: {str(e)}")
            return {
                **state,
                "error": f"感知阶段出错: {str(e)}",
                "current_phase": ResearchPhase.PERCEPTION.value
            }

class ModelingProcessor(PhaseProcessor):
    """建模阶段处理器"""
    
    def process(self, state: TechResearchAgentState) -> TechResearchAgentState:
        """建模阶段：构建技术架构模型"""
        print("2. 建模阶段：构建技术架构模型...")
        
        try:
            # 确保比赛数据已存在
            if not state.get("competition_data"):
                return {
                    **state,
                    "error": "建模阶段缺少比赛数据",
                    "current_phase": ResearchPhase.PERCEPTION.value
                }
            
            # 准备提示
            prompt = ChatPromptTemplate.from_template(PromptTemplates.get_modeling_prompt())
            
            # 构建输入
            input_data = {
                "research_topic": state["research_topic"],
                "competition_focus": state["competition_focus"],
                "time_period": state["time_period"],
                "competition_data": json.dumps(state["competition_data"], ensure_ascii=False, indent=2)
            }
            
            # 调用LLM
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke(input_data)
            
            # 更新状态
            return {
                **state,
                "technical_model": result,
                "current_phase": ResearchPhase.REASONING.value
            }
        except Exception as e:
            logging.error(f"建模阶段出错: {str(e)}")
            return {
                **state,
                "error": f"建模阶段出错: {str(e)}",
                "current_phase": ResearchPhase.MODELING.value
            }

class ReasoningProcessor(PhaseProcessor):
    """推理阶段处理器"""
    
    def process(self, state: TechResearchAgentState) -> TechResearchAgentState:
        """推理阶段：分析具体的技术解决方案"""
        print("3. 推理阶段：分析技术解决方案...")
        
        try:
            # 确保技术模型已存在
            if not state.get("technical_model"):
                return {
                    **state,
                    "error": "推理阶段缺少技术模型",
                    "current_phase": ResearchPhase.MODELING.value
                }
            
            # 准备提示
            prompt = ChatPromptTemplate.from_template(PromptTemplates.get_reasoning_prompt())
            
            # 构建输入
            input_data = {
                "research_topic": state["research_topic"],
                "competition_focus": state["competition_focus"],
                "time_period": state["time_period"],
                "technical_model": json.dumps(state["technical_model"], ensure_ascii=False, indent=2)
            }
            
            # 调用LLM
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke(input_data)
            
            # 更新状态
            return {
                **state,
                "solution_analysis": result,
                "current_phase": ResearchPhase.DECISION.value
            }
        except Exception as e:
            logging.error(f"推理阶段出错: {str(e)}")
            return {
                **state,
                "error": f"推理阶段出错: {str(e)}",
                "current_phase": ResearchPhase.REASONING.value
            }

class DecisionProcessor(PhaseProcessor):
    """决策阶段处理器"""
    
    def process(self, state: TechResearchAgentState) -> TechResearchAgentState:
        """决策阶段：对比分析并选择推荐方案"""
        print("4. 决策阶段：对比分析和推荐...")
        
        try:
            # 确保解决方案分析已存在
            if not state.get("solution_analysis"):
                return {
                    **state,
                    "error": "决策阶段缺少解决方案分析",
                    "current_phase": ResearchPhase.REASONING.value
                }
            
            # 准备提示
            prompt = ChatPromptTemplate.from_template(PromptTemplates.get_decision_prompt())
            
            # 构建输入
            input_data = {
                "research_topic": state["research_topic"],
                "competition_focus": state["competition_focus"],
                "time_period": state["time_period"],
                "technical_model": json.dumps(state["technical_model"], ensure_ascii=False, indent=2),
                "solution_analysis": json.dumps(state["solution_analysis"], ensure_ascii=False, indent=2)
            }
            
            # 调用LLM
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke(input_data)
            
            # 更新状态
            return {
                **state,
                "comparison_results": result,
                "current_phase": ResearchPhase.REPORT.value
            }
        except Exception as e:
            logging.error(f"决策阶段出错: {str(e)}")
            return {
                **state,
                "error": f"决策阶段出错: {str(e)}",
                "current_phase": ResearchPhase.DECISION.value
            }

class ReportProcessor(PhaseProcessor):
    """报告阶段处理器"""
    
    def process(self, state: TechResearchAgentState) -> TechResearchAgentState:
        """报告阶段：生成完整的技术调研报告"""
        print("5. 报告阶段：生成技术调研报告...")
        
        try:
            # 确保对比结果已存在
            if not state.get("comparison_results"):
                return {
                    **state,
                    "error": "报告阶段缺少对比结果",
                    "current_phase": ResearchPhase.DECISION.value
                }
            
            # 准备提示
            prompt = ChatPromptTemplate.from_template(PromptTemplates.get_report_prompt())
            
            # 构建输入
            input_data = {
                "research_topic": state["research_topic"],
                "competition_focus": state["competition_focus"],
                "time_period": state["time_period"],
                "competition_data": json.dumps(state["competition_data"], ensure_ascii=False, indent=2),
                "technical_model": json.dumps(state["technical_model"], ensure_ascii=False, indent=2),
                "solution_analysis": json.dumps(state["solution_analysis"], ensure_ascii=False, indent=2),
                "comparison_results": json.dumps(state["comparison_results"], ensure_ascii=False, indent=2)
            }
            
            # 调用LLM
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke(input_data)
            
            # 更新状态
            return {
                **state,
                "final_report": result,
                "current_phase": ResearchPhase.COMPLETED.value
            }
        except Exception as e:
            logging.error(f"报告生成阶段出错: {str(e)}")
            return {
                **state,
                "error": f"报告生成阶段出错: {str(e)}",
                "current_phase": ResearchPhase.REPORT.value
            }

# ========== 工作流管理 ==========
class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_manager = LLMManager(config)
        self.processors = self._create_processors()
    
    def _create_processors(self) -> Dict[str, PhaseProcessor]:
        """创建阶段处理器"""
        return {
            ResearchPhase.PERCEPTION.value: PerceptionProcessor(self.llm_manager),
            ResearchPhase.MODELING.value: ModelingProcessor(self.llm_manager),
            ResearchPhase.REASONING.value: ReasoningProcessor(self.llm_manager),
            ResearchPhase.DECISION.value: DecisionProcessor(self.llm_manager),
            ResearchPhase.REPORT.value: ReportProcessor(self.llm_manager)
        }
    
    def create_workflow(self) -> StateGraph:
        """创建工作流图"""
        # 创建状态图
        workflow = StateGraph(TechResearchAgentState)
        
        # 添加节点
        for phase, processor in self.processors.items():
            workflow.add_node(phase, processor.process)
        
        # 设置入口点
        workflow.set_entry_point(ResearchPhase.PERCEPTION.value)
        
        # 设置边和转换条件
        phases = list(ResearchPhase)
        for i in range(len(phases) - 1):
            current_phase = phases[i].value
            next_phase = phases[i + 1].value
            if next_phase != ResearchPhase.COMPLETED.value:
                workflow.add_edge(current_phase, next_phase)
            else:
                workflow.add_edge(current_phase, END)
        
        # 编译工作流
        return workflow.compile()

# ========== 报告管理器 ==========
class ReportManager:
    """报告管理器"""
    
    @staticmethod
    def save_report(report_content: str, filename_prefix: str = "tech_research_report") -> str:
        """保存报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        return filename
    
    @staticmethod
    def display_report(report_content: str) -> None:
        """显示报告内容"""
        print("\n=== 技术调研报告 ===\n")
        print(report_content)

# ========== 用户界面管理 ==========
class UserInterface:
    """用户界面管理"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
    
    def display_welcome(self) -> None:
        """显示欢迎信息"""
        print("=== 技术调研智能体 - Agent比赛技术解决方案调研 ===\n")
        print("使用模型：Qwen-Turbo-2025-04-28\n")
    
    def display_preset_topics(self) -> None:
        """显示预设调研主题"""
        print("预设调研主题选项：")
        for key, value in self.config.preset_topics.items():
            print(f"{key}. {value['topic']}")
        print("4. 自定义调研主题")
    
    def get_user_input(self) -> Tuple[str, str, str]:
        """获取用户输入"""
        choice = input("\n请选择调研主题 (1-4): ")
        
        if choice in self.config.preset_topics:
            topic_info = self.config.preset_topics[choice]
            return topic_info["topic"], topic_info["competition"], topic_info["period"]
        elif choice == "4":
            topic = input("请输入调研主题: ")
            competition = input("请输入比赛焦点: ")
            period = input("请输入时间范围: ")
            return topic, competition, period
        else:
            print("无效选择，使用默认主题")
            default_info = self.config.preset_topics["1"]
            return default_info["topic"], default_info["competition"], default_info["period"]
    
    def display_research_info(self, topic: str, competition: str, period: str) -> None:
        """显示调研信息"""
        print(f"\n开始调研：{topic}")
        print(f"比赛焦点：{competition}")
        print(f"时间范围：{period}")
        print("\n技术调研智能体开始工作...\n")

# ========== 主控制器 ==========
class TechResearchAgent:
    """技术调研智能体主控制器"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.workflow_manager = WorkflowManager(self.config)
        self.ui = UserInterface(self.config)
        self.report_manager = ReportManager()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
    
    def run(self, topic: str, competition: str, period: str) -> Dict[str, Any]:
        """运行技术调研智能体"""
        try:
            # 创建工作流
            agent = self.workflow_manager.create_workflow()
            
            # 准备初始状态
            initial_state = {
                "research_topic": topic,
                "competition_focus": competition,
                "time_period": period,
                "competition_data": None,
                "technical_model": None,
                "solution_analysis": None,
                "comparison_results": None,
                "selected_solutions": None,
                "final_report": None,
                "current_phase": ResearchPhase.PERCEPTION.value,
                "error": None
            }
            
            print("LangGraph Mermaid流程图：")
            print(agent.get_graph().draw_mermaid())
            
            # 运行智能体
            result = agent.invoke(initial_state)
            
            return result
            
        except Exception as e:
            logging.error(f"运行智能体时发生错误: {str(e)}")
            raise
    
    def run_interactive(self) -> None:
        """交互式运行"""
        try:
            # 显示欢迎信息
            self.ui.display_welcome()
            
            # 显示预设主题
            self.ui.display_preset_topics()
            
            # 获取用户输入
            topic, competition, period = self.ui.get_user_input()
            
            # 显示调研信息
            self.ui.display_research_info(topic, competition, period)
            
            # 运行智能体
            result = self.run(topic, competition, period)
            
            # 处理结果
            if result.get("error"):
                print(f"\n发生错误: {result['error']}")
            else:
                # 显示报告
                self.report_manager.display_report(result.get("final_report", "未生成报告"))
                
                # 保存报告
                filename = self.report_manager.save_report(
                    result.get("final_report", "未生成报告")
                )
                print(f"\n技术调研报告已保存为: {filename}")
                
        except Exception as e:
            print(f"\n运行过程中发生错误: {str(e)}")

# ========== 主函数 ==========
def main():
    """主函数"""
    agent = TechResearchAgent()
    agent.run_interactive()

if __name__ == "__main__":
    main()
