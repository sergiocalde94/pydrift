class ColumnsNotMatchException(Exception):
    """Custom exception to raise when
    columns doesn't match
    """
    pass


class DriftEstimatorException(Exception):
    """Custom exception to raise the
    estimator exceptions that can be
    because numerical data, categorical
    data or the discriminative model
    """
    pass
