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
- 支持对话记忆功能
- 支持人类反馈机制
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
# 添加记忆和反馈相关导入
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory

# ========== 配置管理 ==========
@dataclass
class AgentConfig:
    """智能体配置类"""
    model_name: str = "Qwen-Turbo-2025-04-28"
    timeout: int = 30
    max_retries: int = 3
    api_key_env: str = "DASHSCOPE_API_KEY"
    
    # 记忆配置
    enable_memory: bool = True
    memory_type: str = "buffer"  # "buffer" 或 "summary"
    max_memory_length: int = 10
    memory_key: str = "chat_history"
    
    # 人类反馈配置
    enable_human_feedback: bool = True
    feedback_threshold: float = 0.7  # 反馈阈值
    auto_continue: bool = False  # 是否自动继续
    feedback_prompt: str = "请对当前结果进行评价 (1-5分，或输入具体建议): "
    
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
    FEEDBACK = "feedback"  # 新增反馈阶段
    COMPLETED = "completed"

class FeedbackType(Enum):
    """反馈类型枚举"""
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"
    CONTINUE = "continue"

# ========== 数据模型 ==========
class HumanFeedback(BaseModel):
    """人类反馈模型"""
    feedback_type: FeedbackType = Field(..., description="反馈类型")
    score: Optional[int] = Field(None, ge=1, le=5, description="评分 (1-5)")
    comment: Optional[str] = Field(None, description="具体建议")
    timestamp: datetime = Field(default_factory=datetime.now, description="反馈时间")
    
    @validator('score')
    def validate_score(cls, v):
        if v is not None and (v < 1 or v > 5):
            raise ValueError('评分必须在1-5之间')
        return v

class FeedbackHistory(BaseModel):
    """反馈历史模型"""
    phase: str = Field(..., description="反馈阶段")
    feedback: HumanFeedback = Field(..., description="反馈内容")
    action_taken: str = Field(..., description="采取的行动")
    improvement: Optional[str] = Field(None, description="改进内容")

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
    
    # 记忆状态
    chat_history: Optional[List[Any]]  # 对话历史
    memory_context: Optional[str]  # 记忆上下文
    
    # 反馈状态
    current_feedback: Optional[HumanFeedback]  # 当前反馈
    feedback_history: Optional[List[FeedbackHistory]]  # 反馈历史
    needs_feedback: bool  # 是否需要反馈
    feedback_context: Optional[str]  # 反馈上下文
    
    # 控制流
    current_phase: Literal["perception", "modeling", "reasoning", "decision", "report", "feedback"]
    error: Optional[str]  # 错误信息

# ========== 记忆管理 ==========
class MemoryManager:
    """记忆管理器"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.memory = self._create_memory()
    
    def _create_memory(self) -> Union[ConversationBufferMemory, ConversationSummaryMemory]:
        """创建记忆实例"""
        if not self.config.enable_memory:
            return None
            
        if self.config.memory_type == "summary":
            return ConversationSummaryMemory(
                memory_key=self.config.memory_key,
                return_messages=True,
                max_token_limit=1000
            )
        else:
            return ConversationBufferMemory(
                memory_key=self.config.memory_key,
                return_messages=True,
                max_token_limit=2000
            )
    
    def add_user_message(self, message: str) -> None:
        """添加用户消息到记忆"""
        if self.memory:
            self.memory.chat_memory.add_user_message(message)
    
    def add_ai_message(self, message: str) -> None:
        """添加AI消息到记忆"""
        if self.memory:
            self.memory.chat_memory.add_ai_message(message)
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """获取记忆变量"""
        if self.memory:
            return self.memory.load_memory_variables({})
        return {}
    
    def clear_memory(self) -> None:
        """清空记忆"""
        if self.memory:
            self.memory.clear()
    
    def save_memory(self, filepath: str) -> None:
        """保存记忆到文件"""
        if self.memory:
            memory_data = {
                "messages": [
                    {"type": msg.__class__.__name__, "content": msg.content}
                    for msg in self.memory.chat_memory.messages
                ],
                "timestamp": datetime.now().isoformat()
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
    
    def load_memory(self, filepath: str) -> None:
        """从文件加载记忆"""
        if self.memory and os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                memory_data = json.load(f)
            
            self.memory.clear()
            for msg_data in memory_data.get("messages", []):
                if msg_data["type"] == "HumanMessage":
                    self.memory.chat_memory.add_user_message(msg_data["content"])
                elif msg_data["type"] == "AIMessage":
                    self.memory.chat_memory.add_ai_message(msg_data["content"])

# ========== 反馈管理 ==========
class FeedbackManager:
    """反馈管理器"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.feedback_history: List[FeedbackHistory] = []
    
    def collect_feedback(self, phase: str, result: str, context: str = "") -> HumanFeedback:
        """收集人类反馈"""
        print(f"\n=== {phase.upper()} 阶段反馈收集 ===")
        print(f"当前结果:\n{result}")
        
        if context:
            print(f"\n上下文:\n{context}")
        
        while True:
            try:
                feedback_input = input(f"\n{self.config.feedback_prompt}")
                
                # 解析反馈输入
                if feedback_input.lower() in ['approve', 'a', '通过', '1']:
                    return HumanFeedback(
                        feedback_type=FeedbackType.APPROVE,
                        score=5,
                        comment="用户批准继续"
                    )
                elif feedback_input.lower() in ['revise', 'r', '修改', '2']:
                    comment = input("请提供修改建议: ")
                    score = int(input("请评分 (1-5): "))
                    return HumanFeedback(
                        feedback_type=FeedbackType.REVISE,
                        score=score,
                        comment=comment
                    )
                elif feedback_input.lower() in ['reject', 'j', '拒绝', '3']:
                    comment = input("请说明拒绝原因: ")
                    score = int(input("请评分 (1-5): "))
                    return HumanFeedback(
                        feedback_type=FeedbackType.REJECT,
                        score=score,
                        comment=comment
                    )
                elif feedback_input.lower() in ['continue', 'c', '继续', '4']:
                    return HumanFeedback(
                        feedback_type=FeedbackType.CONTINUE,
                        score=4,
                        comment="用户选择继续"
                    )
                else:
                    # 尝试解析为数字评分
                    try:
                        score = int(feedback_input)
                        if 1 <= score <= 5:
                            comment = input("请提供具体建议 (可选): ")
                            feedback_type = FeedbackType.APPROVE if score >= 4 else FeedbackType.REVISE
                            return HumanFeedback(
                                feedback_type=feedback_type,
                                score=score,
                                comment=comment
                            )
                    except ValueError:
                        pass
                    
                    print("无效输入，请重新输入")
                    
            except KeyboardInterrupt:
                print("\n用户取消反馈")
                return HumanFeedback(
                    feedback_type=FeedbackType.CONTINUE,
                    score=3,
                    comment="用户取消反馈"
                )
    
    def process_feedback(self, feedback: HumanFeedback, phase: str, current_result: str) -> Tuple[str, str]:
        """处理反馈并返回行动和改进内容"""
        action_taken = ""
        improvement = ""
        
        if feedback.feedback_type == FeedbackType.APPROVE:
            action_taken = "继续下一阶段"
            improvement = "用户批准，无需修改"
        elif feedback.feedback_type == FeedbackType.REVISE:
            action_taken = "根据反馈修改当前阶段"
            improvement = feedback.comment or "根据用户建议进行修改"
        elif feedback.feedback_type == FeedbackType.REJECT:
            action_taken = "重新执行当前阶段"
            improvement = feedback.comment or "根据用户反馈重新开始"
        elif feedback.feedback_type == FeedbackType.CONTINUE:
            action_taken = "继续执行"
            improvement = "用户选择继续"
        
        # 记录反馈历史
        feedback_record = FeedbackHistory(
            phase=phase,
            feedback=feedback,
            action_taken=action_taken,
            improvement=improvement
        )
        self.feedback_history.append(feedback_record)
        
        return action_taken, improvement
    
    def get_feedback_summary(self) -> str:
        """获取反馈摘要"""
        if not self.feedback_history:
            return "暂无反馈记录"
        
        summary = "反馈历史摘要:\n"
        for record in self.feedback_history:
            summary += f"- {record.phase}: {record.feedback.feedback_type.value} "
            summary += f"(评分: {record.feedback.score}/5)\n"
            if record.feedback.comment:
                summary += f"  建议: {record.feedback.comment}\n"
            summary += f"  行动: {record.action_taken}\n\n"
        
        return summary
    
    def save_feedback_history(self, filepath: str) -> None:
        """保存反馈历史"""
        feedback_data = {
            "feedback_history": [
                {
                    "phase": record.phase,
                    "feedback": {
                        "type": record.feedback.feedback_type.value,
                        "score": record.feedback.score,
                        "comment": record.feedback.comment,
                        "timestamp": record.feedback.timestamp.isoformat()
                    },
                    "action_taken": record.action_taken,
                    "improvement": record.improvement
                }
                for record in self.feedback_history
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
    
    def load_feedback_history(self, filepath: str) -> None:
        """加载反馈历史"""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
            
            self.feedback_history = []
            for record_data in feedback_data.get("feedback_history", []):
                feedback = HumanFeedback(
                    feedback_type=FeedbackType(record_data["feedback"]["type"]),
                    score=record_data["feedback"]["score"],
                    comment=record_data["feedback"]["comment"],
                    timestamp=datetime.fromisoformat(record_data["feedback"]["timestamp"])
                )
                
                record = FeedbackHistory(
                    phase=record_data["phase"],
                    feedback=feedback,
                    action_taken=record_data["action_taken"],
                    improvement=record_data.get("improvement")
                )
                self.feedback_history.append(record)

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

{memory_context}

{feedback_context}

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

{memory_context}

{feedback_context}

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

{memory_context}

{feedback_context}

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

{memory_context}

{feedback_context}

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

{memory_context}

{feedback_context}

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

    @staticmethod
    def get_feedback_prompt() -> str:
        """获取反馈处理提示模板"""
        return """你是一个技术调研专家，请根据用户的反馈对当前结果进行改进：

调研主题: {research_topic}
当前阶段: {current_phase}

{memory_context}

用户反馈:
- 类型: {feedback_type}
- 评分: {feedback_score}/5
- 建议: {feedback_comment}

当前结果:
{current_result}

请根据用户反馈对结果进行改进，重点关注：
1. 用户指出的问题
2. 评分较低的原因
3. 具体的改进建议
4. 保持原有优势

输出改进后的结果，格式与当前阶段要求一致。
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
    
    def __init__(self, llm_manager: LLMManager, memory_manager: MemoryManager, feedback_manager: FeedbackManager):
        self.llm_manager = llm_manager
        self.memory_manager = memory_manager
        self.feedback_manager = feedback_manager
        self.llm = llm_manager.get_llm()
    
    def process(self, state: TechResearchAgentState) -> TechResearchAgentState:
        """处理阶段逻辑"""
        raise NotImplementedError
    
    def _get_memory_context(self, state: TechResearchAgentState) -> str:
        """获取记忆上下文"""
        if not self.memory_manager.memory:
            return ""
        
        memory_vars = self.memory_manager.get_memory_variables()
        chat_history = memory_vars.get(self.memory_manager.config.memory_key, [])
        
        if not chat_history:
            return ""
        
        # 构建记忆上下文
        context_parts = []
        for msg in chat_history[-5:]:  # 只取最近5条消息
            if hasattr(msg, 'content'):
                role = "用户" if isinstance(msg, HumanMessage) else "AI"
                context_parts.append(f"{role}: {msg.content}")
        
        if context_parts:
            return f"对话历史:\n" + "\n".join(context_parts) + "\n\n"
        return ""
    
    def _get_feedback_context(self, state: TechResearchAgentState) -> str:
        """获取反馈上下文"""
        if not state.get("feedback_history"):
            return ""
        
        feedback_summary = self.feedback_manager.get_feedback_summary()
        return f"反馈历史:\n{feedback_summary}\n"
    
    def _handle_feedback(self, state: TechResearchAgentState, phase: str, result: str) -> TechResearchAgentState:
        """处理反馈"""
        if not self.llm_manager.config.enable_human_feedback:
            return state
        
        # 收集反馈
        feedback = self.feedback_manager.collect_feedback(phase, result)
        action_taken, improvement = self.feedback_manager.process_feedback(feedback, phase, result)
        
        # 更新状态
        state = {
            **state,
            "current_feedback": feedback,
            "feedback_history": self.feedback_manager.feedback_history,
            "needs_feedback": False
        }
        
        # 根据反馈类型决定下一步行动
        if feedback.feedback_type == FeedbackType.REVISE:
            # 需要修改，重新处理当前阶段
            print(f"根据反馈进行修改: {improvement}")
            return self._revise_with_feedback(state, phase, result, feedback)
        elif feedback.feedback_type == FeedbackType.REJECT:
            # 需要重新开始当前阶段
            print(f"重新开始 {phase} 阶段: {improvement}")
            return state  # 保持当前阶段，重新处理
        else:
            # 批准或继续，进入下一阶段
            print(f"反馈处理完成: {action_taken}")
            return state
    
    def _revise_with_feedback(self, state: TechResearchAgentState, phase: str, result: str, feedback: HumanFeedback) -> TechResearchAgentState:
        """根据反馈修改结果"""
        try:
            # 准备反馈处理提示
            prompt = ChatPromptTemplate.from_template(PromptTemplates.get_feedback_prompt())
            
            # 构建输入
            input_data = {
                "research_topic": state["research_topic"],
                "current_phase": phase,
                "memory_context": self._get_memory_context(state),
                "feedback_type": feedback.feedback_type.value,
                "feedback_score": feedback.score or 0,
                "feedback_comment": feedback.comment or "",
                "current_result": result
            }
            
            # 调用LLM进行改进
            chain = prompt | self.llm | StrOutputParser()
            improved_result = chain.invoke(input_data)
            
            # 更新状态
            return {
                **state,
                "feedback_context": f"已根据反馈改进: {feedback.comment}"
            }
            
        except Exception as e:
            logging.error(f"反馈处理出错: {str(e)}")
            return state

class PerceptionProcessor(PhaseProcessor):
    """感知阶段处理器"""
    
    def process(self, state: TechResearchAgentState) -> TechResearchAgentState:
        """感知阶段：收集和整理比赛信息"""
        print("1. 感知阶段：收集Agent比赛信息...")
        
        try:
            # 准备提示
            prompt = ChatPromptTemplate.from_template(PromptTemplates.get_perception_prompt())
            
            # 获取上下文
            memory_context = self._get_memory_context(state)
            feedback_context = self._get_feedback_context(state)
            
            # 构建输入
            input_data = {
                "research_topic": state["research_topic"],
                "competition_focus": state["competition_focus"],
                "time_period": state["time_period"],
                "memory_context": memory_context,
                "feedback_context": feedback_context
            }
            
            # 调用LLM
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke(input_data)
            
            # 添加对话到记忆
            self.memory_manager.add_user_message(f"调研主题: {state['research_topic']}")
            self.memory_manager.add_ai_message(f"已收集比赛信息: {result.get('competition_name', '未知比赛')}")
            
            # 处理反馈
            if self.llm_manager.config.enable_human_feedback:
                state = self._handle_feedback(state, "perception", json.dumps(result, ensure_ascii=False, indent=2))
            
            # 更新状态
            next_phase = ResearchPhase.MODELING.value
            if state.get("current_feedback") and state["current_feedback"].feedback_type == FeedbackType.REJECT:
                next_phase = ResearchPhase.PERCEPTION.value  # 重新开始当前阶段
            
            return {
                **state,
                "competition_data": result,
                "current_phase": next_phase,
                "chat_history": self.memory_manager.get_memory_variables().get(self.memory_manager.config.memory_key, []),
                "memory_context": memory_context,
                "feedback_context": feedback_context
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
            
            # 获取上下文
            memory_context = self._get_memory_context(state)
            feedback_context = self._get_feedback_context(state)
            
            # 构建输入
            input_data = {
                "research_topic": state["research_topic"],
                "competition_focus": state["competition_focus"],
                "time_period": state["time_period"],
                "memory_context": memory_context,
                "feedback_context": feedback_context,
                "competition_data": json.dumps(state["competition_data"], ensure_ascii=False, indent=2)
            }
            
            # 调用LLM
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke(input_data)
            
            # 添加对话到记忆
            self.memory_manager.add_user_message(f"调研主题: {state['research_topic']}")
            self.memory_manager.add_ai_message(f"已构建技术架构模型: {result.get('dominant_frameworks', ['未知框架'])[0]}")
            
            # 处理反馈
            if self.llm_manager.config.enable_human_feedback:
                state = self._handle_feedback(state, "modeling", json.dumps(result, ensure_ascii=False, indent=2))
            
            # 更新状态
            next_phase = ResearchPhase.REASONING.value
            if state.get("current_feedback") and state["current_feedback"].feedback_type == FeedbackType.REJECT:
                next_phase = ResearchPhase.MODELING.value  # 重新开始当前阶段
            
            return {
                **state,
                "technical_model": result,
                "current_phase": next_phase,
                "chat_history": self.memory_manager.get_memory_variables().get(self.memory_manager.config.memory_key, []),
                "memory_context": memory_context,
                "feedback_context": feedback_context
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
            
            # 获取上下文
            memory_context = self._get_memory_context(state)
            feedback_context = self._get_feedback_context(state)
            
            # 构建输入
            input_data = {
                "research_topic": state["research_topic"],
                "competition_focus": state["competition_focus"],
                "time_period": state["time_period"],
                "memory_context": memory_context,
                "feedback_context": feedback_context,
                "technical_model": json.dumps(state["technical_model"], ensure_ascii=False, indent=2)
            }
            
            # 调用LLM
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke(input_data)
            
            # 添加对话到记忆
            self.memory_manager.add_user_message(f"调研主题: {state['research_topic']}")
            self.memory_manager.add_ai_message(f"已分析技术解决方案: {result[0].get('solution_name', '未知方案')}")
            
            # 处理反馈
            if self.llm_manager.config.enable_human_feedback:
                state = self._handle_feedback(state, "reasoning", json.dumps(result, ensure_ascii=False, indent=2))
            
            # 更新状态
            next_phase = ResearchPhase.DECISION.value
            if state.get("current_feedback") and state["current_feedback"].feedback_type == FeedbackType.REJECT:
                next_phase = ResearchPhase.REASONING.value  # 重新开始当前阶段
            
            return {
                **state,
                "solution_analysis": result,
                "current_phase": next_phase,
                "chat_history": self.memory_manager.get_memory_variables().get(self.memory_manager.config.memory_key, []),
                "memory_context": memory_context,
                "feedback_context": feedback_context
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
            
            # 获取上下文
            memory_context = self._get_memory_context(state)
            feedback_context = self._get_feedback_context(state)
            
            # 构建输入
            input_data = {
                "research_topic": state["research_topic"],
                "competition_focus": state["competition_focus"],
                "time_period": state["time_period"],
                "memory_context": memory_context,
                "feedback_context": feedback_context,
                "technical_model": json.dumps(state["technical_model"], ensure_ascii=False, indent=2),
                "solution_analysis": json.dumps(state["solution_analysis"], ensure_ascii=False, indent=2)
            }
            
            # 调用LLM
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke(input_data)
            
            # 添加对话到记忆
            self.memory_manager.add_user_message(f"调研主题: {state['research_topic']}")
            self.memory_manager.add_ai_message(f"已对比分析解决方案: {result[0].get('solution_a', '未知方案A')} vs {result[0].get('solution_b', '未知方案B')}")
            
            # 处理反馈
            if self.llm_manager.config.enable_human_feedback:
                state = self._handle_feedback(state, "decision", json.dumps(result, ensure_ascii=False, indent=2))
            
            # 更新状态
            next_phase = ResearchPhase.REPORT.value
            if state.get("current_feedback") and state["current_feedback"].feedback_type == FeedbackType.REJECT:
                next_phase = ResearchPhase.DECISION.value  # 重新开始当前阶段
            
            return {
                **state,
                "comparison_results": result,
                "current_phase": next_phase,
                "chat_history": self.memory_manager.get_memory_variables().get(self.memory_manager.config.memory_key, []),
                "memory_context": memory_context,
                "feedback_context": feedback_context
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
            
            # 获取上下文
            memory_context = self._get_memory_context(state)
            feedback_context = self._get_feedback_context(state)
            
            # 构建输入
            input_data = {
                "research_topic": state["research_topic"],
                "competition_focus": state["competition_focus"],
                "time_period": state["time_period"],
                "memory_context": memory_context,
                "feedback_context": feedback_context,
                "competition_data": json.dumps(state["competition_data"], ensure_ascii=False, indent=2),
                "technical_model": json.dumps(state["technical_model"], ensure_ascii=False, indent=2),
                "solution_analysis": json.dumps(state["solution_analysis"], ensure_ascii=False, indent=2),
                "comparison_results": json.dumps(state["comparison_results"], ensure_ascii=False, indent=2)
            }
            
            # 调用LLM
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke(input_data)
            
            # 添加对话到记忆
            self.memory_manager.add_user_message(f"调研主题: {state['research_topic']}")
            self.memory_manager.add_ai_message(f"已生成技术调研报告: {state['research_topic']}")
            
            # 处理反馈
            if self.llm_manager.config.enable_human_feedback:
                state = self._handle_feedback(state, "report", result)
            
            # 更新状态
            next_phase = ResearchPhase.COMPLETED.value
            if state.get("current_feedback") and state["current_feedback"].feedback_type == FeedbackType.REJECT:
                next_phase = ResearchPhase.REPORT.value  # 重新开始当前阶段
            
            return {
                **state,
                "final_report": result,
                "current_phase": next_phase,
                "chat_history": self.memory_manager.get_memory_variables().get(self.memory_manager.config.memory_key, []),
                "memory_context": memory_context,
                "feedback_context": feedback_context
            }
        except Exception as e:
            logging.error(f"报告生成阶段出错: {str(e)}")
            return {
                **state,
                "error": f"报告生成阶段出错: {str(e)}",
                "current_phase": ResearchPhase.REPORT.value
            }

class FeedbackProcessor(PhaseProcessor):
    """反馈阶段处理器"""
    
    def process(self, state: TechResearchAgentState) -> TechResearchAgentState:
        """反馈阶段：处理用户反馈"""
        print("反馈阶段：处理用户反馈...")
        
        try:
            # 获取当前阶段的结果
            current_result = self._get_current_result(state)
            
            # 处理反馈
            state = self._handle_feedback(state, state["current_phase"], current_result)
            
            # 决定下一步
            if state.get("current_feedback"):
                if state["current_feedback"].feedback_type in [FeedbackType.APPROVE, FeedbackType.CONTINUE]:
                    # 继续下一阶段
                    next_phase = self._get_next_phase(state["current_phase"])
                    return {
                        **state,
                        "current_phase": next_phase,
                        "needs_feedback": False
                    }
                else:
                    # 重新处理当前阶段
                    return {
                        **state,
                        "needs_feedback": False
                    }
            
            return state
            
        except Exception as e:
            logging.error(f"反馈阶段出错: {str(e)}")
            return {
                **state,
                "error": f"反馈阶段出错: {str(e)}",
                "current_phase": ResearchPhase.FEEDBACK.value
            }
    
    def _get_current_result(self, state: TechResearchAgentState) -> str:
        """获取当前阶段的结果"""
        phase = state["current_phase"]
        if phase == ResearchPhase.PERCEPTION.value:
            return json.dumps(state.get("competition_data", {}), ensure_ascii=False, indent=2)
        elif phase == ResearchPhase.MODELING.value:
            return json.dumps(state.get("technical_model", {}), ensure_ascii=False, indent=2)
        elif phase == ResearchPhase.REASONING.value:
            return json.dumps(state.get("solution_analysis", []), ensure_ascii=False, indent=2)
        elif phase == ResearchPhase.DECISION.value:
            return json.dumps(state.get("comparison_results", []), ensure_ascii=False, indent=2)
        elif phase == ResearchPhase.REPORT.value:
            return state.get("final_report", "")
        else:
            return "未知阶段结果"
    
    def _get_next_phase(self, current_phase: str) -> str:
        """获取下一阶段"""
        phases = [p.value for p in ResearchPhase]
        try:
            current_index = phases.index(current_phase)
            if current_index < len(phases) - 1:
                return phases[current_index + 1]
        except ValueError:
            pass
        return ResearchPhase.COMPLETED.value

# ========== 工作流管理 ==========
class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_manager = LLMManager(config)
        self.memory_manager = MemoryManager(config)
        self.feedback_manager = FeedbackManager(config)
        self.processors = self._create_processors()
    
    def _create_processors(self) -> Dict[str, PhaseProcessor]:
        """创建阶段处理器"""
        return {
            ResearchPhase.PERCEPTION.value: PerceptionProcessor(self.llm_manager, self.memory_manager, self.feedback_manager),
            ResearchPhase.MODELING.value: ModelingProcessor(self.llm_manager, self.memory_manager, self.feedback_manager),
            ResearchPhase.REASONING.value: ReasoningProcessor(self.llm_manager, self.memory_manager, self.feedback_manager),
            ResearchPhase.DECISION.value: DecisionProcessor(self.llm_manager, self.memory_manager, self.feedback_manager),
            ResearchPhase.REPORT.value: ReportProcessor(self.llm_manager, self.memory_manager, self.feedback_manager),
            ResearchPhase.FEEDBACK.value: FeedbackProcessor(self.llm_manager, self.memory_manager, self.feedback_manager)
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
    
    def get_memory_manager(self) -> MemoryManager:
        """获取记忆管理器"""
        return self.memory_manager
    
    def get_feedback_manager(self) -> FeedbackManager:
        """获取反馈管理器"""
        return self.feedback_manager

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
        self.memory_manager = self.workflow_manager.get_memory_manager()
        self.feedback_manager = self.workflow_manager.get_feedback_manager()
        
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
                "chat_history": [],
                "memory_context": "",
                "current_feedback": None,
                "feedback_history": [],
                "needs_feedback": False,
                "feedback_context": "",
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
    
    # 记忆管理接口
    def add_user_message(self, message: str) -> None:
        """添加用户消息到记忆"""
        self.memory_manager.add_user_message(message)
    
    def add_ai_message(self, message: str) -> None:
        """添加AI消息到记忆"""
        self.memory_manager.add_ai_message(message)
    
    def get_chat_history(self) -> List[Any]:
        """获取对话历史"""
        memory_vars = self.memory_manager.get_memory_variables()
        return memory_vars.get(self.memory_manager.config.memory_key, [])
    
    def clear_memory(self) -> None:
        """清空记忆"""
        self.memory_manager.clear_memory()
    
    def save_memory(self, filepath: str) -> None:
        """保存记忆到文件"""
        self.memory_manager.save_memory(filepath)
    
    def load_memory(self, filepath: str) -> None:
        """从文件加载记忆"""
        self.memory_manager.load_memory(filepath)
    
    # 反馈管理接口
    def get_feedback_summary(self) -> str:
        """获取反馈摘要"""
        return self.feedback_manager.get_feedback_summary()
    
    def save_feedback_history(self, filepath: str) -> None:
        """保存反馈历史"""
        self.feedback_manager.save_feedback_history(filepath)
    
    def load_feedback_history(self, filepath: str) -> None:
        """加载反馈历史"""
        self.feedback_manager.load_feedback_history(filepath)
    
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
                
                # 保存记忆和反馈
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if self.config.enable_memory:
                    memory_filename = f"memory_{timestamp}.json"
                    self.save_memory(memory_filename)
                    print(f"对话记忆已保存为: {memory_filename}")
                
                if self.config.enable_human_feedback:
                    feedback_filename = f"feedback_{timestamp}.json"
                    self.save_feedback_history(feedback_filename)
                    print(f"反馈历史已保存为: {feedback_filename}")
                
        except Exception as e:
            print(f"\n运行过程中发生错误: {str(e)}")

# ========== 主函数 ==========
def main():
    """主函数"""
    # 创建带记忆和反馈功能的配置
    config = AgentConfig(
        enable_memory=True,
        memory_type="buffer",
        enable_human_feedback=True,
        feedback_threshold=0.7,
        auto_continue=False
    )
    
    agent = TechResearchAgent(config)
    agent.run_interactive()

if __name__ == "__main__":
    main()
