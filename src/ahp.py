"""
ahp.py
------
Core implementation of the Analytic Hierarchy Process (AHP) for
multi-criteria decision analysis.

This module implements Saaty's (1980) AHP methodology, including:
    - Pairwise comparison matrix validation
    - Priority vector calculation via the eigenvector method
    - Consistency Index (CI) and Consistency Ratio (CR) computation

Reference:
    Saaty, T.L. (1980). The Analytic Hierarchy Process.
    McGraw-Hill, New York.

Author: Rafi Khandoker
Institution: University of Leeds
"""

import numpy as np


# Saaty's Random Consistency Index table (n = 1 to 10)
# Source: Saaty (1980)
RANDOM_INDEX = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49
}


class AHPMatrix:
    """
    Represents an AHP pairwise comparison matrix and provides methods
    to compute priority weights and assess consistency.

    Parameters
    ----------
    matrix : list of list of float
        An n x n positive reciprocal matrix where entry [i][j] represents
        how much criterion i is preferred over criterion j on Saaty's
        1-9 scale. Entry [j][i] must equal 1 / [i][j].

    Raises
    ------
    ValueError
        If the matrix is not square, not positive, or larger than 10x10.

    Examples
    --------
    >>> matrix = [[1, 3, 5], [1/3, 1, 2], [1/5, 1/2, 1]]
    >>> ahp = AHPMatrix(matrix)
    >>> ahp.consistency_ratio()
    0.0036...
    """

    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float)
        self._validate()
        self.n = self.matrix.shape[0]

    def _validate(self):
        """
        Validates that the matrix is square, positive, and within size limits.

        Raises
        ------
        ValueError
            If any validation check fails.
        """
        rows, cols = self.matrix.shape
        if rows != cols:
            raise ValueError(
                f"Matrix must be square. Got shape ({rows}, {cols})."
            )
        if rows > 10:
            raise ValueError(
                f"Matrix size {rows}x{rows} exceeds maximum supported "
                f"size of 10x10 (Saaty's RI table limit)."
            )
        if np.any(self.matrix <= 0):
            raise ValueError(
                "All matrix entries must be positive. "
                "Use Saaty's 1-9 scale with reciprocals for reverse comparisons."
            )

    def priority_vector(self):
        """
        Computes the priority vector using the eigenvector method.

        The priority vector is the normalised principal eigenvector of the
        pairwise comparison matrix. It represents the relative importance
        (weight) of each criterion.

        Returns
        -------
        numpy.ndarray
            A 1D array of length n containing the priority weights.
            All values sum to 1.0.

        Notes
        -----
        Uses numpy's eigenvalue decomposition. The principal eigenvector
        corresponds to the largest eigenvalue (lambda_max). Only the real
        part is returned, as imaginary components arise from floating-point
        precision and are negligible for valid AHP matrices.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.matrix)

        # Identify the index of the largest (principal) eigenvalue
        max_index = np.argmax(eigenvalues.real)

        # Extract the corresponding eigenvector (real part only)
        principal_eigenvector = eigenvectors[:, max_index].real

        # Normalise so all weights sum to 1
        weights = principal_eigenvector / principal_eigenvector.sum()
        return weights

    def lambda_max(self):
        """
        Returns the principal (largest) eigenvalue of the matrix.

        In a perfectly consistent matrix, lambda_max equals n (the matrix
        size). Deviation from n indicates inconsistency.

        Returns
        -------
        float
            The largest real eigenvalue of the comparison matrix.
        """
        eigenvalues = np.linalg.eigvals(self.matrix)
        return float(np.max(eigenvalues.real))

    def consistency_index(self):
        """
        Computes the Consistency Index (CI).

        CI = (lambda_max - n) / (n - 1)

        A CI of 0 indicates perfect consistency. Larger values indicate
        greater inconsistency in the pairwise judgements.

        Returns
        -------
        float
            The Consistency Index. Returns 0.0 for a 1x1 or 2x2 matrix,
            as consistency is always perfect in these cases.
        """
        if self.n <= 2:
            return 0.0
        return (self.lambda_max() - self.n) / (self.n - 1)

    def consistency_ratio(self):
        """
        Computes the Consistency Ratio (CR).

        CR = CI / RI

        where RI is Saaty's Random Consistency Index for a matrix of size n.
        CR < 0.10 indicates acceptable consistency per Saaty's guidelines.

        Returns
        -------
        float
            The Consistency Ratio.

        Raises
        ------
        ValueError
            If the matrix size is not in Saaty's RI table (n > 10).
        """
        ri = RANDOM_INDEX.get(self.n)
        if ri is None:
            raise ValueError(
                f"No Random Index available for matrix size {self.n}. "
                f"Supported sizes: 1 to 10."
            )
        if ri == 0.0:
            # CR is undefined for n <= 2; consistency is assumed perfect
            return 0.0
        return self.consistency_index() / ri

    def is_consistent(self, threshold=0.1):
        """
        Returns True if the matrix meets Saaty's consistency threshold.

        Parameters
        ----------
        threshold : float, optional
            The maximum acceptable CR value. Saaty's recommended threshold
            is 0.10 (default). Some applications use 0.05 for stricter
            requirements.

        Returns
        -------
        bool
            True if CR <= threshold, False otherwise.
        """
        return self.consistency_ratio() <= threshold

    def summary(self):
        """
        Prints a formatted summary of AHP results to stdout.

        Outputs the priority vector, lambda_max, CI, CR, and a
        consistency verdict.
        """
        pv = self.priority_vector()
        cr = self.consistency_ratio()
        verdict = "CONSISTENT (CR < 0.10)" if self.is_consistent() \
                  else "INCONSISTENT (CR >= 0.10) — revision recommended"

        print("=" * 50)
        print("AHP MATRIX ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Matrix size:         {self.n} x {self.n}")
        print(f"Lambda max:          {self.lambda_max():.4f}")
        print(f"Consistency Index:   {self.consistency_index():.4f}")
        print(f"Random Index (RI):   {RANDOM_INDEX[self.n]}")
        print(f"Consistency Ratio:   {cr:.4f}")
        print(f"Verdict:             {verdict}")
        print()
        print("Priority Weights:")
        for i, w in enumerate(pv):
            print(f"  Criterion {i + 1}:  {w:.4f}  ({w * 100:.1f}%)")
        print("=" * 50)
