EXTRAS_REQUIRE = dict(
    test=["pytest", "pytest-cov", "pytest-score"],
    doc=["sphinx", "sphinx-rtd-theme", "ipython>=6.2"],
    dev=[
        "pylint==2.4.4",
        "pre-commit==2.0.0",
        "prospector==1.2.0",
        "yapf==0.29",
    ],
)
EXTRAS_REQUIRE["dev"] += EXTRAS_REQUIRE["doc"] + EXTRAS_REQUIRE["test"]
