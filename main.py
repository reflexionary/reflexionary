"""
Tethys - Your AI Financial Co-Pilot
Main Application Entry Point

This is the main entry point for Tethys Financial Co-Pilot, providing
a unified interface to all Tethys capabilities including Memory Layer,
Mathematical Intelligence, and business services.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_application():
    """Initialize the Tethys application components."""
    from config.app_settings import (
        LOG_LEVEL, 
        ANNOY_INDEX_DIR,
        GEMINI_API_KEY,
        FIREBASE_SERVICE_ACCOUNT_KEY_PATH
    )
    
    # Set log level from config
    logging.getLogger().setLevel(LOG_LEVEL.upper())
    
    logger.info("Initializing Tethys AI Financial Co-Pilot...")
    
    # Verify required environment variables
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set. Some features may not work.")
    
    if not FIREBASE_SERVICE_ACCOUNT_KEY_PATH or not os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY_PATH):
        logger.warning("FIREBASE_SERVICE_ACCOUNT_KEY_PATH is not set or invalid. Firebase features will be disabled.")
    
    # Ensure required directories exist
    os.makedirs(ANNOY_INDEX_DIR, exist_ok=True)
    
    logger.info("Application initialization complete.")

def run_cli():
    """Run the application in CLI mode."""
    print("\n" + "="*60)
    print("🌊 TETHYS - Your AI Financial Co-Pilot")
    print("="*60)
    print("Powered by Memory Layer + Mathematical Intelligence")
    print("="*60)
    
    # Initialize Tethys Core
    try:
        from tethys_core import get_tethys_core
        tethys = get_tethys_core()
        print("✅ Tethys Core initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Tethys Core: {e}")
        return
    
    # Get system status
    try:
        status = tethys.get_system_status()
        print(f"\n📊 System Status: {status['overall_health'].upper()}")
        print(f"   Healthy Components: {status['healthy_components']}/{status['total_components']}")
        
        for component, health in status['component_health'].items():
            status_icon = "✅" if health['status'] == 'healthy' else "⚠️" if health['status'] == 'degraded' else "❌"
            print(f"   {status_icon} {component.replace('_', ' ').title()}: {health['status']}")
        
    except Exception as e:
        print(f"❌ Failed to get system status: {e}")
    
    print("\n" + "-"*60)
    print("Available Commands:")
    print("  query <user_id> <query>     - Process a user query")
    print("  dashboard <user_id>         - Get user dashboard")
    print("  sync <user_id>              - Sync user data")
    print("  status                      - Show system status")
    print("  help                        - Show this help")
    print("  exit                        - Exit Tethys")
    print("-"*60)
    
    # Interactive CLI loop
    while True:
        try:
            command = input("\n🌊 Tethys> ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == "exit" or cmd == "quit":
                print("👋 Goodbye! Thank you for using Tethys.")
                break
            
            elif cmd == "help":
                print("\nAvailable Commands:")
                print("  query <user_id> <query>     - Process a user query")
                print("  dashboard <user_id>         - Get user dashboard")
                print("  sync <user_id>              - Sync user data")
                print("  status                      - Show system status")
                print("  help                        - Show this help")
                print("  exit                        - Exit Tethys")
            
            elif cmd == "status":
                status = tethys.get_system_status()
                print(f"\n📊 System Status: {status['overall_health'].upper()}")
                print(f"   Healthy Components: {status['healthy_components']}/{status['total_components']}")
                
                for component, health in status['component_health'].items():
                    status_icon = "✅" if health['status'] == 'healthy' else "⚠️" if health['status'] == 'degraded' else "❌"
                    print(f"   {status_icon} {component.replace('_', ' ').title()}: {health['status']}")
                    print(f"      Details: {health['details']}")
            
            elif cmd == "query":
                if len(parts) < 3:
                    print("❌ Usage: query <user_id> <query>")
                    continue
                
                user_id = parts[1]
                query = " ".join(parts[2:])
                
                print(f"\n🤔 Processing query for user {user_id}...")
                result = tethys.process_user_query(user_id, query)
                
                if result['status'] == 'success':
                    print(f"✅ Query processed successfully in {result['context']['processing_time_ms']:.2f}ms")
                    print(f"\n💬 Response: {result['response']}")
                    
                    if result['insights']:
                        print(f"\n💡 Insights ({len(result['insights'])}):")
                        for insight in result['insights']:
                            priority_icon = "🔴" if insight['priority'] == 'high' else "🟡" if insight['priority'] == 'medium' else "🟢"
                            print(f"   {priority_icon} {insight['title']}: {insight['description']}")
                    
                    if result['recommendations']:
                        print(f"\n🎯 Recommendations ({len(result['recommendations'])}):")
                        for rec in result['recommendations']:
                            print(f"   📋 {rec['title']}: {rec['description']}")
                            print(f"      Action: {rec['action']}")
                else:
                    print(f"❌ Query processing failed: {result.get('error', 'Unknown error')}")
            
            elif cmd == "dashboard":
                if len(parts) < 2:
                    print("❌ Usage: dashboard <user_id>")
                    continue
                
                user_id = parts[1]
                print(f"\n📊 Generating dashboard for user {user_id}...")
                result = tethys.get_user_dashboard(user_id)
                
                if result['status'] == 'success':
                    print(f"✅ Dashboard generated successfully in {result['processing_time_ms']:.2f}ms")
                    
                    # Portfolio summary
                    portfolio = result.get('portfolio', {})
                    if portfolio:
                        print(f"\n💼 Portfolio Overview:")
                        performance = portfolio.get('performance_metrics', {})
                        if performance:
                            total_return = performance.get('total_return', 0)
                            print(f"   📈 Total Return: {total_return:.2%}")
                    
                    # Goals summary
                    goals = result.get('goals', {})
                    if goals:
                        print(f"\n🎯 Goals Overview:")
                        total_goals = goals.get('total_goals', 0)
                        progress = goals.get('overall_progress_percentage', 0)
                        print(f"   📋 Total Goals: {total_goals}")
                        print(f"   📊 Overall Progress: {progress:.1f}%")
                    
                    # Alerts summary
                    alerts = result.get('alerts', [])
                    if alerts:
                        print(f"\n⚠️ Recent Alerts ({len(alerts)}):")
                        for alert in alerts[:3]:  # Show first 3
                            print(f"   🔔 {alert.get('title', 'Alert')}")
                    
                    # Insights summary
                    insights = result.get('insights', [])
                    if insights:
                        print(f"\n💡 Insights ({len(insights)}):")
                        for insight in insights[:3]:  # Show first 3
                            priority_icon = "🔴" if insight['priority'] == 'high' else "🟡" if insight['priority'] == 'medium' else "🟢"
                            print(f"   {priority_icon} {insight['title']}")
                else:
                    print(f"❌ Dashboard generation failed: {result.get('error', 'Unknown error')}")
            
            elif cmd == "sync":
                if len(parts) < 2:
                    print("❌ Usage: sync <user_id>")
                    continue
                
                user_id = parts[1]
                print(f"\n🔄 Syncing data for user {user_id}...")
                result = tethys.sync_user_data(user_id)
                
                if result['status'] == 'success':
                    print(f"✅ Data sync completed successfully in {result['processing_time_ms']:.2f}ms")
                    sync_result = result.get('sync_result', {})
                    if sync_result.get('status') == 'success':
                        summary = sync_result.get('sync_summary', {})
                        print(f"   📊 Accounts: {summary.get('accounts_count', 0)}")
                        print(f"   💳 Transactions: {summary.get('transactions_count', 0)}")
                        print(f"   📈 Holdings: {summary.get('holdings_count', 0)}")
                else:
                    print(f"❌ Data sync failed: {result.get('error', 'Unknown error')}")
            
            else:
                print(f"❌ Unknown command: {cmd}")
                print("Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Thank you for using Tethys.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_api_server():
    """Run the Tethys API server."""
    print("\n🚀 Starting Tethys API Server...")
    print("To launch the API server, run: uvicorn ml_ops.model_serving.serve_tethys_models:app --reload --host 0.0.0.0 --port 8000")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Tethys AI Financial Co-Pilot")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli", 
                       help="Run mode: cli (interactive) or api (server)")
    parser.add_argument("--user-id", help="User ID for testing")
    parser.add_argument("--query", help="Query for testing")
    
    args = parser.parse_args()
    
    try:
        initialize_application()
        
        if args.mode == "cli":
            if args.user_id and args.query:
                # Test mode
                from tethys_core import get_tethys_core
                tethys = get_tethys_core()
                print(f"\n🧪 Testing Tethys with user {args.user_id}")
                result = tethys.process_user_query(args.user_id, args.query)
                print(f"Result: {result}")
            else:
                run_cli()
        elif args.mode == "api":
            run_api_server()
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
