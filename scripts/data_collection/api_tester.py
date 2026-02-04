"""
API Connectivity Tester for Financial Data Sources
Tests Yahoo Finance, Alpha Vantage, and FRED APIs.
Week 1 Deliverable for Chooser Option Pricing Project.
"""

import yfinance as yf
import pandas as pd
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root directory
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file, override=False)  # Don't override existing env vars
        _ENV_FILE_LOADED = True
        _ENV_FILE_PATH = env_file
    else:
        # Try to load from current directory as fallback
        load_dotenv(override=False)
        _ENV_FILE_LOADED = False
        _ENV_FILE_PATH = None
except ImportError:
    # python-dotenv not installed, will only use system environment variables
    _ENV_FILE_LOADED = False
    _ENV_FILE_PATH = None

warnings.filterwarnings('ignore')

# Constants
REPORT_BASE_DIR = Path(__file__).parent.parent.parent / 'data' / 'reports' / 'api_tests'
MAX_OLD_REPORTS = 10  # Keep only the most recent N reports


# Helper function to make objects JSON-serializable
def make_serializable(obj):
    """Recursively convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, (pd.Index, pd.DatetimeIndex)):
        return [str(x) for x in obj]
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif isinstance(obj, (int, float)):
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        return obj
    elif not isinstance(obj, (str, bool, type(None))):
        return str(obj)
    return obj

# Try to import optional APIs
try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    print("WARNING: Alpha Vantage not installed. Install with: pip install alpha-vantage")
    ALPHA_VANTAGE_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    print("WARNING: FRED API not installed. Install with: pip install fredapi")
    FRED_AVAILABLE = False


class APITester:
    """Tests connectivity to financial data APIs."""
    
    def __init__(self, api_keys: Optional[Dict] = None, report_dir: Optional[Path] = None):
        """
        Initialize API tester.
        
        Args:
            api_keys: Dictionary containing 'alpha_vantage' and 'fred' keys
            report_dir: Directory to save test reports (default: data/reports/api_tests/)
        """
        self.api_keys = api_keys or {}
        self.test_results = {}
        self.report_dir = report_dir or self._get_default_report_dir()
        self.setup_clients()
    
    @staticmethod
    def _get_default_report_dir() -> Path:
        """Get default report directory with date-based subdirectory."""
        base_dir = REPORT_BASE_DIR
        date_str = datetime.now().strftime('%Y-%m-%d')
        report_dir = base_dir / date_str
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir
    
    def setup_clients(self):
        """Initialize API client connections."""
        print("Initializing API clients...")
        
        # Yahoo Finance (always available, no API key needed)
        self.yf_client = yf
        print("[OK] Yahoo Finance client ready")
        
        # Alpha Vantage
        self.av_client = None
        if ALPHA_VANTAGE_AVAILABLE:
            av_key = self.api_keys.get('alpha_vantage', 'demo')
            try:
                self.av_client = TimeSeries(key=av_key, output_format='pandas')
                print(f"[OK] Alpha Vantage client initialized (key: {'demo' if av_key == 'demo' else 'provided'})")
            except Exception as e:
                print(f"[FAIL] Alpha Vantage client failed: {e}")
        else:
            print("[FAIL] Alpha Vantage library not installed")
        
        # FRED API
        self.fred_client = None
        fred_key = self.api_keys.get('fred')
        if FRED_AVAILABLE:
            if fred_key:
                try:
                    self.fred_client = Fred(api_key=fred_key)
                    print("[OK] FRED API client initialized")
                except Exception as e:
                    print(f"[FAIL] FRED API client failed: {e}")
            else:
                print("[FAIL] FRED API key not provided in .env file")
                print("       Set FRED_API_KEY in .env file to test FRED API")
        else:
            print("[FAIL] FRED API library not installed")
    
    def test_yahoo_finance(self) -> Tuple[bool, Dict]:
        """
        Test Yahoo Finance API with JPM stock data.
        
        Test Logic:
        1. Use Ticker.history() instead of download() to avoid MultiIndex issues
        2. Ticker.history() returns DataFrame with simple column names (Open, High, Low, Close, Volume)
        3. Validate required columns exist
        4. Extract sample data (close prices, dates)
        5. Test additional features (ticker info, VIX index)
        """
        print("\n" + "="*50)
        print("TEST 1: YAHOO FINANCE API")
        print("="*50)
        
        try:
            symbol = "JPM"
            print(f"Fetching {symbol} data for the last 7 days...")
            
            # Use Ticker.history() - returns DataFrame with simple column names, no MultiIndex
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="7d", auto_adjust=True)
            
            if data.empty:
                print("[FAIL] No data received")
                return False, {"error": "Empty dataframe"}
            
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = list(data.columns)
            missing_cols = [col for col in required_columns if col not in available_cols]
            
            if missing_cols:
                print(f"[FAIL] Missing columns: {missing_cols}")
                print(f"  Available: {available_cols}")
                return False, {"error": f"Missing columns: {missing_cols}", "available": available_cols}
            
            # Get additional data
            print("Fetching ticker information and VIX index...")
            info = ticker.info
            vix_data = yf.Ticker("^VIX").history(period="3d")
            
            # Extract sample data
            close_prices = data['Close'].head(3).tolist()
            dates = [d.strftime('%Y-%m-%d') for d in data.index[:3]]
            
            # Prepare result
            result = {
                "success": True,
                "jpm_data_points": len(data),
                "jpm_columns": available_cols,
                "date_range": {
                    "start": str(data.index.min()),
                    "end": str(data.index.max())
                },
                "has_ticker_info": bool(info),
                "vix_data_points": len(vix_data) if not vix_data.empty else 0,
                "sample_data": {
                    "dates": dates,
                    "close_prices": close_prices
                }
            }
            
            print(f"[OK] Success! Retrieved {len(data)} days of {symbol} data")
            print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
            print(f"  Columns: {', '.join(available_cols)}")
            
            return True, result
            
        except Exception as e:
            print(f"[FAIL] Yahoo Finance test failed: {e}")
            return False, {"error": str(e)}
    
    def test_alpha_vantage(self) -> Tuple[bool, Dict]:
        """Test Alpha Vantage API."""
        print("\n" + "="*50)
        print("TEST 2: ALPHA VANTAGE API")
        print("="*50)
        
        if not self.av_client:
            print("[FAIL] Alpha Vantage client not available")
            return False, {"error": "Client not initialized"}
        
        # Check if using demo key
        av_key = self.api_keys.get('alpha_vantage', 'demo')
        if av_key == 'demo':
            print("[WARN] Using demo key - functionality is limited")
            print("  TIP: Get a free API key at https://www.alphavantage.co/support/#api-key")
            return False, {"error": "Demo key not supported for this operation"}
        
        try:
            symbol = "JPM"
            print(f"Fetching {symbol} data from Alpha Vantage...")
            
            data, meta_data = self.av_client.get_daily(symbol=symbol, outputsize='compact')
            
            if data.empty:
                print("[FAIL] No data received")
                return False, {"error": "Empty dataframe"}
            
            # Convert sample data to JSON-serializable format
            serializable_sample = make_serializable(data.head(3).to_dict())
            
            result = {
                "success": True,
                "data_points": len(data),
                "columns": list(data.columns),
                "meta_data": {
                    "symbol": meta_data.get('2. Symbol', 'N/A'),
                    "last_refreshed": meta_data.get('3. Last Refreshed', 'N/A'),
                    "time_zone": meta_data.get('5. Time Zone', 'N/A')
                },
                "sample_data": serializable_sample
            }
            
            print(f"[OK] Success! Retrieved {len(data)} data points")
            print(f"  Symbol: {meta_data.get('2. Symbol', 'N/A')}")
            print(f"  Last Refreshed: {meta_data.get('3. Last Refreshed', 'N/A')}")
            
            # Respect rate limits (free tier: 5 requests per minute)
            print("  Waiting 12 seconds for rate limiting...")
            time.sleep(12)
            
            return True, result
            
        except Exception as e:
            error_msg = str(e)
            print(f"[FAIL] Alpha Vantage test failed: {error_msg}")
            
            if "Invalid API call" in error_msg:
                print("  TIP: Try using the demo key: ALPHA_VANTAGE_KEY=demo")
            elif "premium" in error_msg.lower():
                print("  TIP: Free tier may be limited. Using demo key for testing.")
            
            return False, {"error": error_msg}
    
    def test_fred_api(self) -> Tuple[bool, Dict]:
        """Test FRED API with Treasury rates."""
        print("\n" + "="*50)
        print("TEST 3: FRED API")
        print("="*50)
        
        if not self.fred_client:
            print("[FAIL] FRED API client not available")
            return False, {"error": "Client not initialized"}
        
        try:
            series_id = 'DGS3MO'
            print(f"Fetching {series_id} (3-Month Treasury Rate)...")
            
            data = self.fred_client.get_series(series_id)
            
            if data.empty:
                print("[FAIL] No data received")
                return False, {"error": "Empty series"}
            
            print("Fetching series information...")
            info = self.fred_client.get_series_info(series_id)
            
            # Test additional series
            additional_tests = {}
            for test_id in ['DGS10', 'FEDFUNDS']:
                try:
                    test_data = self.fred_client.get_series(test_id)
                    additional_tests[test_id] = {
                        "data_points": len(test_data),
                        "recent_value": float(test_data.iloc[-1]) if len(test_data) > 0 else None
                    }
                    time.sleep(0.2)
                except Exception as e:
                    additional_tests[test_id] = {"error": str(e)}
            
            # Extract recent values
            recent_values = data.tail().tolist()
            
            result = {
                "success": True,
                "primary_series": series_id,
                "data_points": len(data),
                "series_info": {
                    "title": info.get('title', 'N/A'),
                    "frequency": info.get('frequency', 'N/A'),
                    "units": info.get('units', 'N/A'),
                    "notes": (info.get('notes', 'N/A')[:100] + "...") if info.get('notes') else 'N/A'
                },
                "recent_values": recent_values,
                "additional_series": additional_tests
            }
            
            print(f"[OK] Success! Retrieved {len(data)} data points")
            print(f"  Series: {info.get('title', 'N/A')}")
            print(f"  Frequency: {info.get('frequency', 'N/A')}")
            
            return True, result
            
        except Exception as e:
            error_msg = str(e)
            print(f"[FAIL] FRED API test failed: {error_msg}")
            
            if "API" in error_msg and "key" in error_msg:
                print("  TIP: Please register for FRED API key at:")
                print("     https://fred.stlouisfed.org/docs/api/api_key.html")
            
            return False, {"error": error_msg}
    
    def run_all_tests(self) -> Dict:
        """Run all API tests and return comprehensive results."""
        print("\n" + "="*60)
        print("COMPREHENSIVE API CONNECTIVITY TEST")
        print("="*60)
        print("Project: Advanced Chooser Option Pricing Model")
        print("Week 1: Data Source Design & Initial Collection")
        print("="*60)
        
        start_time = time.time()
        
        self.test_results['yahoo_finance'] = self.test_yahoo_finance()
        self.test_results['alpha_vantage'] = self.test_alpha_vantage()
        self.test_results['fred'] = self.test_fred_api()
        
        duration = time.time() - start_time
        summary = self.generate_summary(duration)
        self.save_test_report(summary)
        self._cleanup_old_reports()
        
        return summary
    
    def generate_summary(self, duration: float) -> Dict:
        """Generate test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "tests": {},
            "overall_status": "PASS",
            "recommendations": []
        }
        
        passed_tests = 0
        total_tests = 0
        
        for api_name, (success, details) in self.test_results.items():
            total_tests += 1
            if success:
                passed_tests += 1
            
            status = "PASS" if success else "FAIL"
            summary["tests"][api_name] = {
                "status": status,
                "success": success,
                "details": details
            }
            
            print(f"{'[OK]' if success else '[FAIL]'} {api_name.replace('_', ' ').title():20} {status}")
            if success and 'data_points' in details:
                print(f"    Data points: {details['data_points']}")
        
        summary["tests_passed"] = passed_tests
        summary["tests_total"] = total_tests
        summary["pass_rate"] = round((passed_tests / total_tests) * 100, 1)
        
        summary["overall_status"] = "PASS" if passed_tests == total_tests else "FAIL"
        status_msg = f"[OK] All {total_tests} tests passed!" if passed_tests == total_tests else f"[WARN] {passed_tests}/{total_tests} tests passed"
        print(f"\n{status_msg}")
        
        print(f"\nTotal duration: {duration:.2f} seconds")
        
        summary["recommendations"] = self.generate_recommendations()
        
        if summary["recommendations"]:
            print("\nRECOMMENDATIONS:")
            for rec in summary["recommendations"]:
                print(f"  - {rec}")
        
        return summary
    
    def generate_recommendations(self) -> list:
        """Generate recommendations based on test results."""
        recommendations = []
        yf_success, _ = self.test_results.get('yahoo_finance', (False, {}))
        av_success, av_details = self.test_results.get('alpha_vantage', (False, {}))
        fred_success, _ = self.test_results.get('fred', (False, {}))
        
        if not yf_success:
            recommendations.append("Check internet connection for Yahoo Finance")
        if not av_success:
            if "Invalid API call" in str(av_details.get('error', '')):
                recommendations.append("Get Alpha Vantage API key: https://www.alphavantage.co/support/#api-key")
            recommendations.append("Or use demo key temporarily: ALPHA_VANTAGE_KEY=demo")
        if not fred_success:
            recommendations.append("Register for FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html")
            recommendations.append("FRED data is optional but recommended for Treasury rates")
        
        if all([yf_success, av_success, fred_success]):
            recommendations.append("All APIs working! Proceed with data collection.")
        elif yf_success:
            recommendations.append("At least Yahoo Finance works. Can proceed with basic data collection.")
        
        return recommendations
    
    def save_test_report(self, summary: Dict) -> Path:
        """Save test results to JSON and text files."""
        # Ensure report directory exists
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"api_test_report_{timestamp}"
        
        # Save JSON report
        json_file = self.report_dir / f"{base_name}.json"
        with open(json_file, 'w') as f:
            json.dump(make_serializable(summary), f, indent=2)
        
        print(f"\nDetailed report saved to: {json_file}")
        
        # Save text summary
        txt_file = self.report_dir / f"{base_name}.txt"
        with open(txt_file, 'w') as f:
            f.write("API CONNECTIVITY TEST REPORT\n" + "="*50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {summary['duration_seconds']} seconds\n")
            f.write(f"Overall Status: {summary['overall_status']}\n")
            f.write(f"Tests Passed: {summary['tests_passed']}/{summary['tests_total']}\n\n")
            f.write("DETAILED RESULTS:\n" + "-"*50 + "\n")
            
            for api_name, test_info in summary['tests'].items():
                f.write(f"\n{api_name.upper().replace('_', ' ')}:\n")
                f.write(f"  Status: {test_info['status']}\n")
                if test_info['success'] and 'data_points' in test_info['details']:
                    f.write(f"  Data Points: {test_info['details']['data_points']}\n")
            
            if summary['recommendations']:
                f.write("\nRECOMMENDATIONS:\n" + "-"*50 + "\n")
                f.write('\n'.join(f"- {rec}" for rec in summary['recommendations']))
        
        print(f"Text summary saved to: {txt_file}")
        
        return json_file
    
    def _cleanup_old_reports(self):
        """Remove old report files, keeping only the most recent N reports."""
        try:
            # Get all report files sorted by modification time
            report_files = sorted(
                self.report_dir.glob("api_test_report_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Remove files beyond the limit
            if len(report_files) > MAX_OLD_REPORTS:
                for old_file in report_files[MAX_OLD_REPORTS:]:
                    # Also remove corresponding .txt file
                    txt_file = old_file.with_suffix('.txt')
                    if txt_file.exists():
                        txt_file.unlink()
                    old_file.unlink()
                    print(f"  Cleaned up old report: {old_file.name}")
        except Exception as e:
            # Don't fail if cleanup fails
            print(f"  [WARN] Could not clean up old reports: {e}")


def load_api_keys() -> Dict[str, str]:
    """Load API keys from environment variables or .env file."""
    api_keys = {}
    
    # Helper to load a single key
    def load_key(env_var: str, key_name: str, default=None, required=False):
        key = os.getenv(env_var)
        
        # Check if key is valid (not empty, not placeholder)
        invalid_values = ['', 'your_key_here', 'your_alpha_vantage_key_here', 'your_fred_api_key_here']
        
        if key and key.strip() and key.strip() not in invalid_values:
            api_keys[key_name] = key.strip()
            print(f"[INFO] {key_name.replace('_', ' ').title()} API key loaded")
            return True
        else:
            if default:
                api_keys[key_name] = default
                print(f"WARNING: {env_var} not found or invalid, using {default}")
            elif required:
                print(f"WARNING: {env_var} not found or invalid, tests may fail")
            else:
                # For FRED, show warning if key is missing
                if key_name == 'fred':
                    print(f"WARNING: {env_var} not found or invalid")
                    print(f"         FRED API tests will be skipped")
            return False
    
    load_key('ALPHA_VANTAGE_KEY', 'alpha_vantage', default='demo')
    load_key('FRED_API_KEY', 'fred')
    
    return api_keys


def main():
    """Main function to run API tests."""
    print("="*60)
    print("API Connectivity Tester")
    print("="*60)
    
    # Show .env file status
    if _ENV_FILE_LOADED:
        print(f"[INFO] Loaded .env file from: {_ENV_FILE_PATH}")
    elif _ENV_FILE_PATH is None:
        print("[INFO] python-dotenv not available, using system environment variables only")
    else:
        print(f"[INFO] .env file not found, using system environment variables only")
    
    print("Loading API keys...\n")
    api_keys = load_api_keys()
    print()
    
    # Create and run tester
    tester = APITester(api_keys)
    summary = tester.run_all_tests()
    
    # Exit with appropriate code
    status_msg = "[OK] API testing completed successfully!" if summary['overall_status'] == 'PASS' else "[WARN] API testing completed with issues\n   Check recommendations above"
    print(f"\n{status_msg}")
    sys.exit(0 if summary['overall_status'] == 'PASS' else 1)


if __name__ == "__main__":
    main()
