use crate::common::runtime_test_support::runtime;

#[test]
fn should_run_affine_four_bit_quantized_matmul_with_group_size_sixty_four() {
    let runtime = runtime();
    let activation_values = (1..=64)
        .map(|activation_index| activation_index as f32)
        .collect::<Vec<_>>();
    let activations = runtime
        .array_from_f32(&activation_values, &[1, 64])
        .expect("the activation matrix should be valid");
    let quantized_weight_words = [
        0x1111_1111,
        0x1111_1111,
        0x1111_1111,
        0x1111_1111,
        0x1111_1111,
        0x1111_1111,
        0x1111_1111,
        0x1111_1111,
        0x2222_2222,
        0x2222_2222,
        0x2222_2222,
        0x2222_2222,
        0x2222_2222,
        0x2222_2222,
        0x2222_2222,
        0x2222_2222,
    ];
    let quantized_weights = runtime
        .array_from_u32(&quantized_weight_words, &[2, 8])
        .expect("the packed 4-bit weight matrix should be valid");
    let scales = runtime
        .array_from_f32(&[1.0, 1.0], &[2, 1])
        .expect("the affine scales should be valid");
    let biases = runtime
        .array_from_f32(&[0.0, 0.0], &[2, 1])
        .expect("the affine biases should be valid");

    let quantized_product = runtime
        .quantized_matmul_affine(
            &activations,
            &quantized_weights,
            &scales,
            &biases,
            true,
            64,
            4,
        )
        .expect("affine 4-bit quantized matmul should build a valid graph");

    assert_eq!(quantized_product.shape(), vec![1, 2]);
    assert_eq!(
        quantized_product
            .to_vec_f32()
            .expect("the quantized product should evaluate as float32"),
        vec![2080.0, 4160.0]
    );
}

#[test]
fn should_run_affine_eight_bit_quantized_matmul_with_group_size_sixty_four() {
    let runtime = runtime();
    let activation_values = (1..=64)
        .map(|activation_index| activation_index as f32)
        .collect::<Vec<_>>();
    let activations = runtime
        .array_from_f32(&activation_values, &[1, 64])
        .expect("the activation matrix should be valid");
    let quantized_weight_words = [
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0101_0101,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
        0x0202_0202,
    ];
    let quantized_weights = runtime
        .array_from_u32(&quantized_weight_words, &[2, 16])
        .expect("the packed 8-bit weight matrix should be valid");
    let scales = runtime
        .array_from_f32(&[1.0, 1.0], &[2, 1])
        .expect("the affine scales should be valid");
    let biases = runtime
        .array_from_f32(&[0.0, 0.0], &[2, 1])
        .expect("the affine biases should be valid");

    let quantized_product = runtime
        .quantized_matmul_affine(
            &activations,
            &quantized_weights,
            &scales,
            &biases,
            true,
            64,
            8,
        )
        .expect("affine 8-bit quantized matmul should build a valid graph");

    assert_eq!(quantized_product.shape(), vec![1, 2]);
    assert_eq!(
        quantized_product
            .to_vec_f32()
            .expect("the quantized product should evaluate as float32"),
        vec![2080.0, 4160.0]
    );
}

#[test]
fn should_run_affine_five_bit_quantized_matmul_with_group_size_thirty_two() {
    let runtime = runtime();
    let activations = runtime
        .array_from_f32(&[1.0; 64], &[1, 64])
        .expect("the activation matrix should be valid");
    let quantized_weights = runtime
        .array_from_u32(&[0; 20], &[2, 10])
        .expect("the packed 5-bit weight matrix should be valid");
    let scales = runtime
        .array_from_f32(&[1.0; 4], &[2, 2])
        .expect("the affine scales should be valid");
    let biases = runtime
        .array_from_f32(&[0.0; 4], &[2, 2])
        .expect("the affine biases should be valid");

    let quantized_product = runtime
        .quantized_matmul_affine(
            &activations,
            &quantized_weights,
            &scales,
            &biases,
            true,
            32,
            5,
        )
        .expect("MLX-supported affine quantization parameters should build a valid graph");

    assert_eq!(quantized_product.shape(), vec![1, 2]);
    assert_eq!(
        quantized_product
            .to_vec_f32()
            .expect("the quantized product should evaluate as float32"),
        vec![0.0, 0.0]
    );
}

#[test]
fn should_run_affine_gather_qmm_for_selected_four_bit_experts() {
    let runtime = runtime();
    let activation_values = (1..=64)
        .map(|activation_index| activation_index as f32)
        .collect::<Vec<_>>();
    let activations = runtime
        .array_from_f32(&activation_values, &[1, 64])
        .expect("the activation matrix should be valid");
    let mut quantized_weight_words = Vec::new();
    quantized_weight_words.extend([0x1111_1111; 8]);
    quantized_weight_words.extend([0x2222_2222; 8]);
    quantized_weight_words.extend([0x5555_5555; 8]);
    quantized_weight_words.extend([0x6666_6666; 8]);
    quantized_weight_words.extend([0x3333_3333; 8]);
    quantized_weight_words.extend([0x4444_4444; 8]);
    let quantized_weights = runtime
        .array_from_u32(&quantized_weight_words, &[3, 2, 8])
        .expect("the packed expert weight tensor should be valid");
    let scales = runtime
        .array_from_f32(&[1.0; 6], &[3, 2, 1])
        .expect("the affine expert scales should be valid");
    let biases = runtime
        .array_from_f32(&[0.0; 6], &[3, 2, 1])
        .expect("the affine expert biases should be valid");
    let selected_expert_indices = runtime
        .array_from_i32(&[2, 0], &[2])
        .expect("the selected expert indices should be valid");

    let selected_expert_products = runtime
        .gather_quantized_matmul_affine(
            &activations,
            &quantized_weights,
            &scales,
            &biases,
            None,
            Some(&selected_expert_indices),
            true,
            64,
            4,
            false,
        )
        .expect("affine gather_qmm should build a valid selected-expert graph");

    assert_eq!(selected_expert_products.shape(), vec![2, 1, 2]);
    assert_eq!(
        selected_expert_products
            .to_vec_f32()
            .expect("the selected expert products should evaluate as float32"),
        vec![6240.0, 8320.0, 2080.0, 4160.0]
    );
}

#[test]
fn should_dequantize_only_selected_eight_bit_embedding_rows() {
    let runtime = runtime();
    let mut quantized_embedding_words = Vec::new();
    quantized_embedding_words.extend([0x0101_0101; 16]);
    quantized_embedding_words.extend([0x0202_0202; 16]);
    let quantized_embeddings = runtime
        .array_from_u32(&quantized_embedding_words, &[2, 16])
        .expect("the packed embedding rows should be valid");
    let embedding_scales = runtime
        .array_from_f32(&[1.0, 1.0], &[2, 1])
        .expect("the embedding scales should be valid");
    let embedding_biases = runtime
        .array_from_f32(&[0.0, 0.0], &[2, 1])
        .expect("the embedding biases should be valid");
    let token_indices = runtime
        .array_from_i32(&[1, 0], &[1, 2])
        .expect("the token indices should be valid");
    let selected_embeddings = runtime
        .take_axis(&quantized_embeddings, &token_indices, 0)
        .expect("only requested packed embedding rows should be selected");
    let selected_scales = runtime
        .take_axis(&embedding_scales, &token_indices, 0)
        .expect("only requested embedding scales should be selected");
    let selected_biases = runtime
        .take_axis(&embedding_biases, &token_indices, 0)
        .expect("only requested embedding biases should be selected");

    let dequantized_embeddings = runtime
        .dequantize_affine(
            &selected_embeddings,
            &selected_scales,
            &selected_biases,
            64,
            8,
        )
        .expect("selected embedding rows should build a dequantization graph");

    assert_eq!(dequantized_embeddings.shape(), vec![1, 2, 64]);
    let actual_embeddings = dequantized_embeddings
        .to_vec_f32()
        .expect("the selected embeddings should evaluate as float32");
    assert_eq!(actual_embeddings[..64], [2.0; 64]);
    assert_eq!(actual_embeddings[64..], [1.0; 64]);
}

#[test]
fn should_multiply_selected_dense_experts_without_materializing_selected_weight_copies() {
    let runtime = runtime();
    let activations = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], &[2, 1, 4])
        .expect("the selected-expert activations should be valid");
    let transposed_expert_weights = runtime
        .array_from_f32(
            &[
                1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0,
                3.0, 0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0,
            ],
            &[3, 4, 2],
        )
        .expect("the dense expert weights should be valid");
    let selected_expert_indices = runtime
        .array_from_i32(&[2, 0], &[2])
        .expect("the selected expert indices should be valid");

    let selected_expert_products = runtime
        .gather_dense_matmul(
            &activations,
            &transposed_expert_weights,
            None,
            Some(&selected_expert_indices),
            false,
        )
        .expect("dense gather_mm should build a selected-expert graph");

    assert_eq!(selected_expert_products.shape(), vec![2, 1, 2]);
    assert_eq!(
        selected_expert_products
            .to_vec_f32()
            .expect("the selected dense expert products should evaluate"),
        vec![15.0, 15.0, 7.0, 7.0]
    );
}
