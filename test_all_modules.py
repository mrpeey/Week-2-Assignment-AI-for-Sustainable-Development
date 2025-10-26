"""
Quick test script to verify all modules are working
Run: python test_all_modules.py
"""

import sys
import traceback

def test_module(name, import_path, test_func=None):
    """Test a single module"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    try:
        module = __import__(import_path, fromlist=[''])
        print(f"✅ Import successful: {import_path}")
        
        if test_func:
            test_func(module)
            print(f"✅ Functionality test passed")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

def test_market_intelligence(module):
    """Test market intelligence module"""
    system = module.MarketIntelligenceSystem()
    news_data = module.generate_sample_news_data()[:2]
    price_data = module.generate_sample_price_data(days=30)
    report = system.generate_market_report(news_data, price_data, 'test_crop')
    assert 'sentiment_analysis' in report
    assert 'price_prediction' in report
    print(f"   Generated report for {report['commodity']}")

def test_smart_irrigation(module):
    """Test smart irrigation module"""
    controller = module.SmartIrrigationController()
    conditions = {
        'soil_moisture': 35,
        'temperature': 28,
        'rainfall_forecast': 2,
        'days_since_planting': 45,
        'growth_stage': 0.4,
        'evapotranspiration': 6.5,
        'cumulative_water': 150,
        'humidity': 60
    }
    rec = controller.get_irrigation_recommendation(conditions)
    assert 'recommended_irrigation_mm' in rec
    assert 'urgency' in rec
    print(f"   Recommendation: {rec['recommended_irrigation_mm']} mm, Urgency: {rec['urgency']}")

def test_yield_prediction(module):
    """Test yield prediction module"""
    system = module.YieldPredictionSystem()
    data = system.create_synthetic_data(n_samples=500)
    print(f"   Created synthetic data: {len(data)} samples")
    # Don't train fully, just verify structure
    assert 'yield_tons_per_hectare' in data.columns
    assert len(data) == 500

def test_crop_disease(module):
    """Test crop disease detection module"""
    detector = module.CropDiseaseDetector()
    model = detector.build_model()
    print(f"   Model initialized successfully")
    assert detector.model is not None

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SMARTFARM AI - MODULE VERIFICATION TEST")
    print("="*60)
    
    results = {}
    
    # Test core modules
    results['Market Intelligence'] = test_module(
        'Market Intelligence',
        'src.market_intelligence',
        test_market_intelligence
    )
    
    results['Smart Irrigation'] = test_module(
        'Smart Irrigation',
        'src.smart_irrigation',
        test_smart_irrigation
    )
    
    results['Yield Prediction'] = test_module(
        'Yield Prediction',
        'src.yield_prediction',
        test_yield_prediction
    )
    
    results['Crop Disease Detection'] = test_module(
        'Crop Disease Detection',
        'src.crop_disease_detection',
        test_crop_disease
    )
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:12} {name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} modules passed")
    
    if passed == total:
        print("\n🎉 All modules are working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} module(s) need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
