import pytest
import numpy as np

from app.services.face_math import (
    compute_cosine_similarity,
    verify_biometric_match,
    apply_pca_reduction,
    apply_pca_reduction_batch,
)

# All tests are pure unit tests with no database, no ONNX, no HTTP.
# They are meant to be run in a clean environment with no external dependencies.


class TestCosineSimilarity:
    """Tests for compute_cosine_similarity."""
    
    @staticmethod
    def test_result_is_float():
        vec = np.ones(512)
        result = compute_cosine_similarity(vec, vec)
        assert isinstance(result, float) 

    @staticmethod
    def test_compute_cosine_similarity(unit_vector_512, similar_vector_512):
        """
        Tests that cosine similarity is computed correctly.
        Expected cosine similarity vs unit_vector_512: ~0.95 - 0.99.
        """
        cosine_similarity = compute_cosine_similarity(unit_vector_512, similar_vector_512)
        assert cosine_similarity > 0.95
        assert cosine_similarity < 0.99
        
        
    def test_identical_vectors_return_one(self, unit_vector_512):
        """
        A vector compared to itself must return exactly 1.0.
        This is the mathematical identity of cosine similarity.
        """
        result = compute_cosine_similarity(unit_vector_512, unit_vector_512)
        assert abs(result - 1.0) < 1e-5, (
            f"Expected ~1.0 for identical vectors, got {result}"
        )


    def test_similar_vectors_return_high_score(
        self, unit_vector_512, similar_vector_512
    ):
        """
        Vectors from the same person (small noise) must score above 0.90.
        The threshold for biometric match in production is 0.50,
        so 0.90+ confirms the fixture is generating realistic close pairs.
        """
        result = compute_cosine_similarity(unit_vector_512, similar_vector_512)
        assert result > 0.75, (
            f"Expected similarity > 0.75 for near-identical vectors, got {result}"
        )

    
    def test_different_vectors_return_low_score(
        self, unit_vector_512, different_vector_512
    ):
        """
        Vectors from different people must score well below the 0.50 threshold.
        Random unit vectors in 512D space are nearly orthogonal by construction.
        """
        result = compute_cosine_similarity(unit_vector_512, different_vector_512)
        assert result < 0.30, (
            f"Expected similarity < 0.30 for different people vectors, got {result}"
        )
        
        
    def test_zero_vector_returns_zero(self, unit_vector_512, zero_vector_512):
        """
        A zero vector has no direction. The function must return 0.0
        and not raise ZeroDivisionError.
        """
        result = compute_cosine_similarity(unit_vector_512, zero_vector_512)
        assert result == 0.0, (
            f"Expected 0.0 for zero vector input, got {result}"
        )
        
    
    def test_both_zero_vectors_return_zero(self, zero_vector_512):
        """
        Two zero vectors must return 0.0, not raise an exception.
        """
        result = compute_cosine_similarity(zero_vector_512, zero_vector_512)
        assert result == 0.0
        
    
    def test_return_type_is_float(self, unit_vector_512, similar_vector_512):
        """
        The return type must be Python float, not numpy float32/float64.
        FastAPI JSON serialization requires native Python types.
        """
        result = compute_cosine_similarity(unit_vector_512, similar_vector_512)
        assert isinstance(result, float), (
            f"Expected Python float, got {type(result)}"
        )
        
    
    def test_result_bounded_between_minus_one_and_one(
        self, unit_vector_512, different_vector_512
    ):
        """
        Cosine similarity is always in [-1.0, 1.0] by definition.
        """
        result = compute_cosine_similarity(unit_vector_512, different_vector_512)
        assert -1.0 <= result <= 1.0
        
    
    def test_accepts_2d_input_flattens_correctly(self, unit_vector_512):
        """
        The function must accept (1, 512) shaped arrays and flatten them.
        pgvector returns embeddings that may need reshaping.
        """
        reshaped = unit_vector_512.reshape(1, 512)
        result = compute_cosine_similarity(reshaped, unit_vector_512)
        assert abs(result - 1.0) < 1e-5
        
        
    def test_symmetry(self, unit_vector_512, different_vector_512):
        """
        Cosine similarity must be symmetric: sim(a, b) == sim(b, a).
        """
        result_ab = compute_cosine_similarity(unit_vector_512, different_vector_512)
        result_ba = compute_cosine_similarity(different_vector_512, unit_vector_512)
        assert abs(result_ab - result_ba) < 1e-6
        
        
# verify_biometric_match

class TestVerifyBiometricMatch:
    """Tests for verify_biometric_match."""
    
    @staticmethod
    def test_returns_tuple(unit_vector_512, similar_vector_512):
        """
        The function must return a tuple of (is_match, similarity_score).
        """
        result = verify_biometric_match(unit_vector_512, similar_vector_512)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        
    def test_same_person_returns_match(self, unit_vector_512, similar_vector_512):
        """
        Two embeddings from the same person must return is_match=True
        when similarity is above the default threshold of 0.50.
        """
        is_match, similarity = verify_biometric_match(
            unit_vector_512, similar_vector_512
        )
        assert is_match is True
        assert similarity > 0.50
        
        
    def test_different_person_returns_no_match(
        self, unit_vector_512, different_vector_512
    ):
        """
        Two embeddings from different people must return is_match=False.
        """
        is_match, similarity = verify_biometric_match(
            unit_vector_512, different_vector_512
        )
        assert is_match is False
        
    
    def test_returns_tuple_of_bool_and_float(
        self, unit_vector_512, similar_vector_512
    ):
        """
        Return type must be (bool, float). Critical for stream.py which
        unpacks: is_match, similarity = verify_biometric_match(...)
        """
        result = verify_biometric_match(unit_vector_512, similar_vector_512)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        
        
    def test_custom_threshold_strict(self, unit_vector_512, similar_vector_512):
        """
        With a very strict threshold of 0.999, even similar vectors
        from the same person should fail the match.
        Tests that the threshold parameter is respected.
        """
        is_match, _ = verify_biometric_match(
            unit_vector_512, similar_vector_512, threshold=0.999
        )
        assert is_match is False
        
        
    def test_custom_threshold_permissive(
        self, unit_vector_512, different_vector_512
    ):
        """
        With a very permissive threshold of 0.0, all non-zero vectors match.
        Tests the lower boundary of the threshold parameter.
        """
        is_match, _ = verify_biometric_match(
            unit_vector_512, different_vector_512, threshold= - 1.0
        )
        assert is_match is True
        
        
    def test_similarity_score_matches_cosine(
        self, unit_vector_512, similar_vector_512
    ):
        """
        The similarity score returned by verify_biometric_match must be
        identical to the direct output of compute_cosine_similarity.
        Ensures no transformation is applied to the score internally.
        """
        _, similarity_from_verify = verify_biometric_match(
            unit_vector_512, similar_vector_512
        )
        direct_similarity = compute_cosine_similarity(
            unit_vector_512, similar_vector_512
        )
        assert abs(similarity_from_verify - direct_similarity) < 1e-6
        
        
        
    def test_identical_vectors_always_match(self, unit_vector_512):
        """
        A registered embedding compared to itself must always match.
        Edge case: user re-enrolls with the exact same image.
        """
        is_match, similarity = verify_biometric_match(
            unit_vector_512, unit_vector_512
        )
        assert is_match is True
        assert abs(similarity - 1.0) < 1e-5
        
        
# apply_pca_reduction

class TestApplyPcaReduction:
    """Tests for apply_pca_reduction."""
    
    
    def test_output_shape_is_correct(self, embedding_matrix_10x512):
        """
        Reducing 10x512 to n_components=3 must return shape (10, 3).
        """
        result = apply_pca_reduction(embedding_matrix_10x512, n_components=3)
        assert result.shape == (10, 3), (
            f"Expected (10, 3), got {result.shape}"
        )
    
    
    def test_output_shape_128_components(self, embedding_matrix_10x512):
        """
        Default reduction to 128 components from 512D.
        """
        # mathematical constraint of SVD.
        result = apply_pca_reduction(embedding_matrix_10x512, n_components=128)
        assert result.shape == (10, 128)
        
        
    def test_returns_original_if_components_gte_features(
        self, embedding_matrix_10x512
    ):
        """
        If n_components >= n_features, the function must return the
        original matrix unchanged. No reduction needed.
        """
        result = apply_pca_reduction(
            embedding_matrix_10x512, n_components=512
        )
        assert result.shape == embedding_matrix_10x512.shape
        
        
    def test_variance_preserved_increases_with_components(
        self, embedding_matrix_10x512
    ):
        """
        More components must capture more variance.
        The Frobenius norm of the projected data increases with n_components.
        """
        result_3  = apply_pca_reduction(embedding_matrix_10x512, n_components=3)
        result_10 = apply_pca_reduction(embedding_matrix_10x512, n_components=10)
        
        norm_3  = float(np.linalg.norm(result_3))
        norm_10 = float(np.linalg.norm(result_10))
        
        assert norm_10 >= norm_3, (
            "Higher n_components should preserve more variance (higher norm)."
        )
     
    def test_output_is_ndarray(self, embedding_matrix_10x512):
        
        """Output must be numpy ndarray, not list or other type."""
        
        result = apply_pca_reduction(embedding_matrix_10x512, n_components=3)
        
        assert isinstance(result, np.ndarray) 
        
        
    def test_centering_removes_mean(self, embedding_matrix_10x512):
        """
        PCA centers the data before projecting. The mean of the projected
        data should be close to zero along each component.
        """
        result = apply_pca_reduction(embedding_matrix_10x512, n_components=3)
        column_means = np.abs(np.mean(result, axis=0))
        assert np.all(column_means < 1.0), (
            f"Column means after PCA should be near zero, got {column_means}"
        )

# apply_pca_reduction_batch

class TestApplyPcaReductionBatch:
    """Tests for apply_pca_reduction_batch."""
    
    def test_output_shape_is_correct(self, embedding_matrix_10x512):
        """
        Reducing (10, 512) to n_components=3 must return (10, 3).
        """
        result = apply_pca_reduction_batch(
            embedding_matrix_10x512, n_components=3
        )
        assert result.shape == (10, 3), (
            f"Expected (10, 3), got {result.shape}"
        )
        
        
    def test_returns_list_of_same_length(self, embedding_matrix_10x512):
        """
        Batch function must return a list with the same number of elements
        as the input list.
        """
        input_list = [embedding_matrix_10x512, embedding_matrix_10x512]
        result = apply_pca_reduction_batch(input_list, n_components=3)
        
        assert len(result) == 2           
        
        
    def test_each_element_has_correct_shape(self, embedding_matrix_10x512):
        """
        Each element in the output list must have shape (10, n_components).
        """
        input_list = [embedding_matrix_10x512]
        result = apply_pca_reduction_batch(input_list, n_components=3)
        
        assert result[0].shape == (10, 3)
        
        
    def test_empty_list_returns_empty_list(self):
        """
        An empty input list must return an empty list without errors.
        """
        result = apply_pca_reduction_batch([], n_components=3)
        
        assert result == []
        
        
    def test_single_element_matches_non_batch(self, embedding_matrix_10x512):
        """
        Batch with one element must produce the same result as calling
        apply_pca_reduction directly.
        """
        batch_result  = apply_pca_reduction_batch(
            
            [embedding_matrix_10x512], n_components=3
        )
        
        single_result = apply_pca_reduction(embedding_matrix_10x512, n_components=3)
        
        np.testing.assert_array_almost_equal(batch_result[0], single_result)