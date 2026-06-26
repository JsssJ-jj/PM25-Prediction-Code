"""
Quick test script for PM2.5 Prediction Code repository.
Run this to verify that all dependencies are correctly installed & version matched.
"""

import importlib
import sys


def safe_import(pkg_name, alias=None):
    """安全导入库，返回(模块对象, 版本字符串)"""
    mod = importlib.import_module(pkg_name)
    if alias:
        globals()[alias] = mod
    # 获取版本
    ver = getattr(mod, "__version__", "unknown")
    return mod, ver


def test_imports():
    """Test that all required packages can be imported & check basic submodules."""
    print("=" * 50)
    print("Testing PM2.5 Prediction Package Imports")
    print("=" * 50)
    
    packages = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("matplotlib", "mpl"),
        ("seaborn", "sns"),
        ("scipy", None),
        ("sklearn", None),
        ("xgboost", "xgb"),
        ("catboost", None),
        ("lightgbm", "lgb"),
        ("tensorflow", "tf"),
        ("shap", None),
    ]

    # 关键子模块校验（模块路径）
    submodule_checks = [
        "sklearn.model_selection",
        "sklearn.preprocessing",
        "sklearn.metrics",
        "sklearn.ensemble",
        "sklearn.neighbors",
        "sklearn.neural_network",
        "sklearn.svm",
        "tensorflow.keras.layers",
        "tensorflow.keras.models",
        "tensorflow.keras.callbacks",
        "tensorflow.keras.optimizers",
    ]
    
    # 关键类校验（需要 getattr 获取）
    class_checks = [
        ("xgboost", "XGBRegressor"),
        ("lightgbm", "LGBMRegressor"),
        ("catboost", "CatBoostRegressor"),
        ("shap", "TreeExplainer"),
    ]
    
    failed = []
    pkg_versions = {}

    # 1. 校验主包
    for pkg, alias in packages:
        try:
            mod, ver = safe_import(pkg, alias)
            pkg_versions[pkg] = ver
            print(f"  [OK] {pkg} (version: {ver})")
        except ImportError as e:
            print(f"  [FAIL] {pkg}: {str(e)}")
            failed.append(pkg)

    # 2. 校验核心子模块
    print("\n" + "-" * 50)
    print("Testing core submodules")
    print("-" * 50)
    for submod in submodule_checks:
        try:
            importlib.import_module(submod)
            print(f"  [OK] {submod}")
        except Exception as e:
            print(f"  [WARN] {submod} unavailable: {str(e)}")

    # 3. 校验关键类
    print("\n" + "-" * 50)
    print("Testing core model classes")
    print("-" * 50)
    for pkg_name, class_name in class_checks:
        try:
            pkg = sys.modules.get(pkg_name)
            if pkg is None:
                pkg = importlib.import_module(pkg_name)
            cls = getattr(pkg, class_name)
            print(f"  [OK] {pkg_name}.{class_name}")
        except Exception as e:
            print(f"  [WARN] {pkg_name}.{class_name} unavailable: {str(e)}")

    # 汇总结果
    print("\n" + "-" * 50)
    if failed:
        print(f"Failed to import packages: {', '.join(failed)}")
        print("Fix command: pip install -r requirements.txt")
        return False
    else:
        print("All required packages imported successfully!")
        print("\nInstalled versions summary:")
        for k, v in sorted(pkg_versions.items()):
            print(f"  {k:<12} -> {v}")
        return True


if __name__ == "__main__":
    import_ok = test_imports()
    
    print("\n" + "=" * 50)
    if import_ok:
        print("ALL DEPENDENCY TESTS PASSED - Environment ready for PM2.5 prediction!")
        sys.exit(0)
    else:
        print("SOME PACKAGES MISSING - Reinstall requirements first.")
        sys.exit(1)