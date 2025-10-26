"""
SmartFarm AI - All-In-One Demo Runner
Runs all 4 core modules sequentially to demonstrate functionality
"""

import subprocess
import sys
import time

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_module(name, command):
    """Run a single module"""
    print_header(f"Running: {name}")
    print(f"Command: {command}\n")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ {name} completed successfully")
            return True
        else:
            print(f"\n❌ {name} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n❌ Error running {name}: {e}")
        return False

def main():
    """Run all demos"""
    print_header("SMARTFARM AI - COMPREHENSIVE DEMO")
    print("This will run all 4 AI modules to demonstrate functionality\n")
    print("Modules:")
    print("  1. Market Intelligence (NLP)")
    print("  2. Smart Irrigation (RL)")
    print("  3. Yield Prediction (ML)")
    print("  4. Crop Disease Detection (CNN)")
    print("\nPress Ctrl+C at any time to stop\n")
    
    input("Press Enter to start...")
    
    python_exe = sys.executable
    
    demos = [
        ("Market Intelligence", f"{python_exe} src/market_intelligence.py --commodity rice --articles 2 --days 60"),
        ("Smart Irrigation", f"{python_exe} src/smart_irrigation.py"),
        ("Yield Prediction", f"{python_exe} src/yield_prediction.py"),
        ("Crop Disease Detection", f"{python_exe} src/crop_disease_detection.py"),
    ]
    
    results = {}
    
    for name, command in demos:
        success = run_module(name, command)
        results[name] = success
        time.sleep(2)  # Brief pause between modules
    
    # Summary
    print_header("DEMO SUMMARY")
    
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status:15} {name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nCompleted: {passed}/{total} modules ran successfully")
    
    if passed == total:
        print("\n🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("  • Run dashboard: run_dashboard.bat")
        print("  • Start API: run_api.bat")
        print("  • See SETUP_COMPLETE.md for details")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
