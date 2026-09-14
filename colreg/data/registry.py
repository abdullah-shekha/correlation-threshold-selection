"""The dataset corpus.

Selection criteria, which belong verbatim in Section 3.1 of the paper:

1.  Public and permanently citable -- UCI ML Repository or OpenML, no
    Kaggle-only re-uploads whose provenance cannot be checked.
2.  Genuine regression target, continuous, not a discretised class.
3.  Predominantly numeric features (categoricals one-hot encoded where few);
    pairwise Pearson correlation must be meaningful for the pruning rule to be
    defined at all.
4.  5 <= p <= 120 and 100 <= n <= 25,000 -- small enough that collinearity
    actually bites, large enough for 10x5 nested CV to be stable.
5.  Spanning the collinearity range by design.  A corpus of well-conditioned
    datasets would answer RQ1 with a trivial null.

``EXPECTED_STRATUM`` is the *a priori* expectation only, recorded here so it can
be compared against the measured condition number.  The stratum used in the
paper is the measured one from ``characterise()`` -- never this column.

The high-collinearity entries are not incidental.  Auto MPG has four engine-size
proxies mutually above r = 0.85; Abalone has seven shell measurements above
0.9; Energy Efficiency contains a pair at r = -0.99 (relative compactness and
surface area are algebraically linked); Parkinsons telemonitoring carries whole
families of jitter and shimmer variants that are near-duplicates by
construction.  These are the datasets where the pruning question is decided.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CORPUS", "DatasetSpec", "by_stratum"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    source: str            # 'openml' | 'uci' | 'sklearn'
    ref: str               # OpenML data name, UCI id, or sklearn loader
    n: int
    p: int
    target: str
    expected_stratum: str  # 'low' | 'medium' | 'high'
    note: str = ""


CORPUS: tuple[DatasetSpec, ...] = (
    # ---------------------------------------------------------- high collinearity
    DatasetSpec("auto_mpg", "Auto MPG", "uci", "9", 398, 7, "mpg", "high",
                "cylinders/displacement/horsepower/weight mutually > 0.85"),
    DatasetSpec("abalone", "Abalone", "uci", "1", 4177, 8, "rings", "high",
                "seven shell measurements mutually > 0.9"),
    DatasetSpec("energy_y1", "Energy Efficiency (heating)", "uci", "242", 768, 8, "Y1", "high",
                "relative compactness vs surface area r = -0.99, algebraic"),
    DatasetSpec("energy_y2", "Energy Efficiency (cooling)", "uci", "242", 768, 8, "Y2", "high",
                "same design matrix, second target"),
    DatasetSpec("parkinsons_motor", "Parkinsons Telemonitoring", "uci", "189", 5875, 19,
                "motor_UPDRS", "high", "jitter and shimmer families are near-duplicates"),
    DatasetSpec("appliances", "Appliances Energy Prediction", "uci", "374", 19735, 26,
                "Appliances", "high", "room temperature/humidity sensor pairs"),
    DatasetSpec("bike_hour", "Bike Sharing (hourly)", "uci", "275", 17379, 12, "cnt", "high",
                "temp vs atemp r = 0.99"),
    DatasetSpec("superconduct", "Superconductivity", "uci", "464", 21263, 81, "critical_temp",
                "high", "engineered aggregates of the same base quantities"),

    # -------------------------------------------------------- medium collinearity
    DatasetSpec("concrete", "Concrete Compressive Strength", "uci", "165", 1030, 8,
                "strength", "medium"),
    DatasetSpec("diabetes", "Diabetes", "sklearn", "load_diabetes", 442, 10, "target", "medium",
                "s1/s2 serum measurements correlated; canonical small benchmark"),
    DatasetSpec("real_estate", "Real Estate Valuation", "uci", "477", 414, 6,
                "house_price_unit_area", "medium"),
    DatasetSpec("automobile", "Automobile", "uci", "10", 205, 15, "price", "medium",
                "UCI declares 'symboling' as target; we predict price, which sits in features"),
    DatasetSpec("cpu_perf", "Computer Hardware", "uci", "29", 209, 6, "PRP", "medium",
                "cache and memory bounds correlated; small n"),
    DatasetSpec("student_por", "Student Performance (Portuguese)", "uci", "320", 649, 30,
                "G3", "medium", "G1/G2 prior grades strongly predict G3"),
    DatasetSpec("communities", "Communities and Crime", "uci", "183", 1994, 100,
                "ViolentCrimesPerPop", "medium", "wide; many demographic rates co-move"),
    DatasetSpec("gas_turbine", "Gas Turbine CO and NOx Emission", "uci", "551", 36733, 9,
                "TEY", "medium", "subsample to 20k for tractability"),
    DatasetSpec("california", "California Housing", "sklearn", "fetch_california_housing",
                20640, 8, "MedHouseVal", "medium", "AveRooms/AveBedrms correlated"),

    # ----------------------------------------------------------- low collinearity
    DatasetSpec("airfoil", "Airfoil Self-Noise", "uci", "291", 1503, 5,
                "scaled_sound_pressure", "low"),
    DatasetSpec("servo", "Servo", "uci", "87", 167, 4, "class", "low", "smallest n in corpus"),
    DatasetSpec("forest_fires", "Forest Fires", "uci", "162", 517, 12, "area", "low",
                "very hard target; useful as a low-signal control"),
    DatasetSpec("istanbul_se", "Istanbul Stock Exchange", "uci", "247", 536, 8, "ISE", "low"),
    DatasetSpec("garment", "Garment Employee Productivity", "uci", "597", 1197, 14,
                "actual_productivity", "low"),
    DatasetSpec("solar_flare", "Solar Flare", "uci", "89", 1066, 10, "c_class", "low"),
    DatasetSpec("facebook", "Facebook Metrics", "uci", "368", 500, 7, "Total_Interactions",
                "low"),
)

# ---------------------------------------------------------------------------
# NOTE FOR THE AUTHOR
#
# The `ref` values are UCI repository ids and are the least stable thing in this
# file.  Before the first real run, execute scripts/verify_corpus.py, which
# fetches every entry, records the resolved URL, the actual (n, p), the SHA-256
# of the raw file, and the measured condition number into data/corpus_manifest.
# Cite that manifest in the paper, not this module.  Reviewers at statistics
# venues do check whether the reported n matches the repository.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NOT IN THE CORPUS: hosted by UCI but not exposed to the Python API.
# fetch_ucirepo raises DatasetNotFoundError ("exists in the repository, but is
# not available for import") for each. They are recorded here so the exclusion
# is a documented decision rather than a silent gap -- a corpus that shrinks
# toward whatever downloaded cleanly is a selection effect on the result.
#
#   Concrete Slump Test   (id 182)  smallest n; mixture components sum to a constant
#   Residential Building  (id 437)  wide, p=103, cost indices heavily correlated
#   QSAR Aquatic Toxicity (id 505)
#   Yacht Hydrodynamics   (id 243)  near-orthogonal by experimental design
#
# Yacht is the one worth replacing: it was the cleanest low-collinearity
# control. To add it, download the .data file by hand from its UCI page
# and add a loader; otherwise the remaining low-stratum entries carry that role.
# ---------------------------------------------------------------------------

def by_stratum(stratum: str) -> tuple[DatasetSpec, ...]:
    return tuple(d for d in CORPUS if d.expected_stratum == stratum)


# Candidates that were characterised but not adopted into the corpus, with the
# inclusion criterion each fails. Kept in code rather than deleted so that the
# corpus is reproducible as a decision, not merely as a list: anyone
# regenerating the manifest gets the same 24 datasets and can see what was
# considered alongside them.
#
# Criterion: each design matrix must represent a SINGLE population. A matrix
# that pools two populations has a correlation structure which is a mixture of
# two structures, and correlation structure is this study's object of interest.
NOT_ADOPTED: tuple[tuple[str, str], ...] = (
    ("wine_quality",
     "UCI id 186 serves red and white wine as one file, so the design matrix "
     "pools two chemically distinct populations. Fails the single-population "
     "criterion. Set aside before any result for it was computed."),
)
