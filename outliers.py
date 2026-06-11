from sklearn.base import BaseEstimator, TransformerMixin

class outliers_handler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.bounds = {}

    def fit(self, X, y=None):

        numeric_cols = X.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            self.bounds[col] = (lower, upper)

        return self

    def transform(self, X):

        X = X.copy()

        # 🔥 always recompute numeric columns (NO STATE DEPENDENCY)
        numeric_cols = X.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            if col in self.bounds:
                lower, upper = self.bounds[col]
                X[col] = X[col].clip(lower, upper)

        return X

    def get_feature_names_out(self, input_features=None):
        return input_features