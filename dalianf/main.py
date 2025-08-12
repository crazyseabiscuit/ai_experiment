#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Agent System Entry Point
多智能体系统入口点

This is the main entry point for the multi-agent system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the multi_agent_system to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

def print_welcome():
    """Print welcome message"""
    print("🤖 Multi-Agent System with Memory, Tools, and Human Feedback")
    print("=" * 70)
    print("A comprehensive multi-agent system integrating:")
    print("• Memory management (short-term, long-term, working)")
    print("• Dynamic tool integration and registry")
    print("• Human feedback mechanisms with learning")
    print("• Multi-agent coordination and communication")
    print("• RAGAS validation and quality assurance")
    print("=" * 70)

async def main():
    """Main entry point"""
    print_welcome()
    
    print("\n🎯 Choose your experience:")
    print("1. 🎭 Comprehensive Demo (Recommended)")
    print("2. 💬 Interactive CLI Interface")
    print("3. 🔥 Stress Test")
    print("4. 📖 System Documentation")
    print("5. 🚪 Exit")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            # Run comprehensive demo
            from multi_agent_demo import MultiAgentDemo
            demo = MultiAgentDemo()
            await demo.run_comprehensive_demo()
            
        elif choice == "2":
            # Run interactive CLI
            from multi_agent_system.cli import MultiAgentCLI
            cli = MultiAgentCLI()
            await cli.initialize_system()
            await cli.interactive_loop()
            
        elif choice == "3":
            # Run stress test
            from multi_agent_demo import MultiAgentDemo
            demo = MultiAgentDemo()
            await demo.run_stress_test()
            
        elif choice == "4":
            # Show documentation
            print("\n📖 System Documentation")
            print("-" * 50)
            print("The multi-agent system consists of:")
            print("\n📁 Core Components:")
            print("• core_system.py - Core architecture and utilities")
            print("• agents.py - Agent implementations")
            print("• tools.py - Tool integration system")
            print("• feedback_system.py - Human feedback mechanisms")
            print("• cli.py - Command-line interface")
            print("\n🎮 Demo Files:")
            print("• multi_agent_demo.py - Comprehensive demonstrations")
            print("• README.md - Complete documentation")
            print("\n🔧 Configuration:")
            print("• Edit core_system.py to set your API key")
            print("• Customize agent prompts in agents.py")
            print("• Add new tools in tools.py")
            print("\n🚀 Getting Started:")
            print("1. Set your DashScope API key")
            print("2. Install dependencies: pip install -r requirements.txt")
            print("3. Run: python main.py")
            print("4. Choose option 1 for the guided demo")
            
        elif choice == "5":
            print("\n👋 Goodbye! Thanks for exploring the multi-agent system!")
            
        else:
            print("\n❌ Invalid choice. Please select 1-5.")
            
    except KeyboardInterrupt:
        print("\n\n👋 System interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
