#!/usr/bin/env python3
"""
Demo script to showcase V2 features:
1. Semantic QA Detection (混合式边界判定)
2. Chain Prompt Processing (超长答案处理)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.semantic_qa_detector import SemanticQADetector
from src.core.chain_prompt_processor import ChainPromptProcessor
from src.core.llm_client import LLMClient


def demo_semantic_detection():
    """展示语义QA检测功能"""
    print("="*80)
    print("🎯 Demo 1: 语义QA检测 - 识别无明显前缀的问答对")
    print("="*80)
    
    # 示例文本：包含隐含的问答对
    test_text = """
段永平最近在投资论坛上分享了他的投资理念。

价值投资还有效吗？

我认为价值投资永远不会过时。价值投资的本质是买入被低估的优秀公司，
这个逻辑在任何时代都是成立的。关键是要有耐心，要能够忍受短期的波动。

网友：您如何看待当前的科技股泡沫？

段永平：泡沫总是会破的，但优秀的公司会存活下来。我不会因为担心泡沫
而错过真正的好公司。重要的是要分清什么是真正的价值，什么是炒作。

投资中最难的是什么？

克服人性的弱点。恐惧和贪婪是投资的两大敌人。当市场疯狂时要保持冷静，
当市场恐慌时要敢于出手。这说起来容易，做起来很难。
"""
    
    # 初始化语义检测器
    try:
        detector = SemanticQADetector()
        
        # 执行检测
        result = detector.showcase_demo(test_text)
        
        print("\n📊 检测结果：")
        print(f"规则检测到的QA对: {len(result['rule_based_pairs'])} 个")
        print(f"语义检测到的QA对: {len(result['semantic_pairs'])} 个")
        print(f"总计QA对: {len(result['all_pairs'])} 个")
        
        print("\n📝 详细结果：")
        paragraphs = result['paragraphs']
        
        # 显示规则检测结果
        print("\n✅ 规则检测（传统方法）：")
        for start, end in result['rule_based_pairs']:
            print(f"  Q: {paragraphs[start]}")
            print(f"  A: {' '.join(paragraphs[start+1:end+1])}\n")
        
        # 显示语义检测结果
        print("\n🚀 语义检测（新增能力）：")
        for start, end in result['semantic_pairs']:
            print(f"  Q: {paragraphs[start]}")
            print(f"  A: {' '.join(paragraphs[start+1:end+1])}\n")
        
        print(result['showcase_message'])
        
    except Exception as e:
        print(f"⚠️ 语义检测器初始化失败: {e}")
        print("提示：需要安装 sentence-transformers 库")
        print("运行: pip install sentence-transformers")


def demo_chain_prompt():
    """展示链式Prompt处理功能"""
    print("\n" + "="*80)
    print("🔗 Demo 2: 链式Prompt处理 - 处理超长答案")
    print("="*80)
    
    # 模拟超长答案
    long_question = "什么是价值投资？"
    long_answer = """价值投资是一种投资策略，它的核心理念是寻找被市场低估的优质公司股票，
并长期持有等待价值回归。这个概念最早由本杰明·格雷厄姆提出，后来被他的学生
沃伦·巴菲特发扬光大。

价值投资的基本原则包括：

1. 安全边际原则：永远不要支付超过内在价值的价格。格雷厄姆认为，投资者应该
在股票价格显著低于其内在价值时买入，这个差额就是安全边际。安全边际越大，
投资的风险就越小。

2. 长期投资：价值投资者通常持有股票数年甚至数十年。他们相信，短期内市场
可能是非理性的，但长期来看，股票价格终将反映公司的真实价值。

3. 基本面分析：价值投资者会深入研究公司的财务报表、商业模式、竞争优势、
管理团队等基本面因素。他们寻找的是具有持续竞争优势的公司。

4. 逆向思维：当大多数人恐慌时买入，当大多数人贪婪时卖出。价值投资者往往
在市场低迷时寻找机会。

5. 集中投资：与分散投资不同，价值投资者倾向于集中投资于少数他们深入了解
的优质公司。巴菲特曾说："分散投资是无知者的自我保护。"

价值投资在实践中的应用：

首先，投资者需要学会如何评估公司的内在价值。这包括分析公司的盈利能力、
成长潜力、资产质量等。常用的估值方法包括市盈率、市净率、现金流折现等。

其次，要有耐心。价值投资不是快速致富的方法。市场认识到一家公司的真实
价值可能需要很长时间。在这个过程中，投资者需要忍受股价的波动，甚至是
账面上的亏损。

第三，要不断学习。市场在变化，行业在进化，投资者需要不断更新自己的知识，
理解新的商业模式和技术趋势。

第四，要有独立思考的能力。不要盲目跟风，要基于自己的研究做出判断。

最后，要控制情绪。投资中最大的敌人往往是自己的情绪。恐惧会让你在底部
卖出，贪婪会让你在顶部买入。

价值投资的成功案例很多。巴菲特通过价值投资成为世界首富，他的伯克希尔·
哈撒韦公司在过去50多年里创造了惊人的回报。在中国，也有很多成功的价值
投资者，他们通过长期持有优质公司股票获得了丰厚的回报。

总的来说，价值投资是一种经过时间检验的投资哲学。它不仅是一种投资方法，
更是一种思维方式。它教会我们要有长远眼光，要深入研究，要独立思考，要
控制情绪。这些原则不仅适用于投资，也适用于生活的其他方面。"""
    
    try:
        # 初始化LLM客户端
        llm_client = LLMClient(
            host="http://localhost:11434",
            model_name="qwen2.5:7b-instruct"
        )
        
        # 初始化链式处理器
        processor = ChainPromptProcessor(
            llm_client=llm_client,
            max_answer_length=500,  # 为了演示，设置较小的阈值
            chunk_size=400,
            summary_length=30
        )
        
        print(f"\n📏 原始答案长度: {len(long_answer)} 字符")
        print("📝 开始链式处理...\n")
        
        # 展示处理流程
        print(processor.showcase_ultra_long_handling())
        
        # 实际处理（如果LLM可用）
        if llm_client._test_connection():
            result = processor.process_long_qa(long_question, long_answer)
            
            print(f"\n✅ 处理完成！")
            print(f"处理方式: {result.get('processing_method')}")
            print(f"处理块数: {result.get('chunks_processed', 0)}")
            print(f"最终答案长度: {len(result.get('answer', ''))}")
        else:
            print("\n⚠️ 无法连接到LLM服务，仅展示处理流程")
            
    except Exception as e:
        print(f"\n❌ 链式处理演示失败: {e}")


def main():
    """主函数"""
    print("🚀 Legend QA Extractor V2 功能展示")
    print("="*80)
    print("本演示展示两个核心改进：")
    print("1. 混合式边界判定（规则+语义）")
    print("2. 超长答案的链式Prompt处理")
    print("="*80)
    
    # Demo 1: 语义检测
    demo_semantic_detection()
    
    # Demo 2: 链式处理
    demo_chain_prompt()
    
    print("\n" + "="*80)
    print("🎯 技术亮点总结：")
    print("="*80)
    print("""
1. 混合式QA检测：
   - 保留高效的规则匹配作为基线
   - 新增语义相似度检测，提升泛化能力
   - 成功识别无明显前缀的隐含问答对
   
2. 链式Prompt处理：
   - 自动识别超长答案并分段处理
   - 使用摘要保持上下文连贯性
   - 解决Context Window限制问题
   
3. 工程实现优势：
   - 向后兼容，可选启用新功能
   - 优雅降级，依赖缺失时自动回退
   - 详细的监控和日志支持
    """)


if __name__ == "__main__":
    main()