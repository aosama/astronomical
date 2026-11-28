use super::{
    AstroSsdPromptCacheHostTensor, AstroSsdPromptCacheHostTensorElements,
    AstroSsdPromptCacheHostTensors, host_tensors_payload_byte_count,
};

fn host_tensor(elements: AstroSsdPromptCacheHostTensorElements) -> AstroSsdPromptCacheHostTensor {
    AstroSsdPromptCacheHostTensor {
        safetensors_dtype_name: "BF16",
        shape: vec![2, 2],
        elements,
    }
}

#[test]
fn should_count_two_bytes_an_element_for_sixteen_bit_state() {
    let tensor = host_tensor(AstroSsdPromptCacheHostTensorElements::SixteenBit(vec![
        1, 2, 3, 4,
    ]));
    assert_eq!(tensor.payload_byte_count(), 8);
    assert_eq!(tensor.element_count(), 4);
}

#[test]
fn should_count_four_bytes_an_element_for_thirty_two_bit_state() {
    let float_tensor = host_tensor(AstroSsdPromptCacheHostTensorElements::Float32(vec![
        1.0, 2.0,
    ]));
    let integer_tensor = host_tensor(AstroSsdPromptCacheHostTensorElements::Int32(vec![7]));
    assert_eq!(float_tensor.payload_byte_count(), 8);
    assert_eq!(integer_tensor.payload_byte_count(), 4);
}

#[test]
fn should_append_little_endian_bytes_for_the_requested_elements_only() {
    let tensor = host_tensor(AstroSsdPromptCacheHostTensorElements::SixteenBit(vec![
        0x0102, 0x0304, 0x0506,
    ]));
    let mut payload_bytes = Vec::new();
    tensor.append_little_endian_bytes(1..3, &mut payload_bytes);
    assert_eq!(payload_bytes, vec![0x04, 0x03, 0x06, 0x05]);
}

#[test]
fn should_append_nothing_for_an_empty_element_range() {
    let tensor = host_tensor(AstroSsdPromptCacheHostTensorElements::Float32(vec![1.5]));
    let mut payload_bytes = vec![9];
    tensor.append_little_endian_bytes(1..1, &mut payload_bytes);
    assert_eq!(payload_bytes, vec![9]);
}

#[test]
fn should_sum_the_payload_of_every_tensor_and_zero_for_none() {
    let mut host_tensors = AstroSsdPromptCacheHostTensors::new();
    assert_eq!(host_tensors_payload_byte_count(&host_tensors), 0);
    host_tensors.insert(
        "keys".to_owned(),
        host_tensor(AstroSsdPromptCacheHostTensorElements::SixteenBit(vec![
            0;
            3
        ])),
    );
    host_tensors.insert(
        "state".to_owned(),
        host_tensor(AstroSsdPromptCacheHostTensorElements::Float32(vec![0.0; 2])),
    );
    assert_eq!(host_tensors_payload_byte_count(&host_tensors), 14);
}
