//! Unit contracts for the restricted pickle reader, on pickles assembled
//! here opcode by opcode in the form `torch.save` writes them.

use super::{PickledStorage, PickledTensor, PickledValue, StorageElementType, read_pickle};
use crate::model_files::test_support::PickleBuilder;

fn head_like_pickle() -> Vec<u8> {
    PickleBuilder::default()
        .opcode(b'}')
        .memo_put(0)
        .opcode(b'(')
        .text("head")
        .global("collections", "OrderedDict")
        .memo_put(1)
        .opcode(b')')
        .opcode(b'R')
        .opcode(b'(')
        .text("q.weight")
        .float32_matrix("0", 4, 8)
        .opcode(b'u')
        .opcode(b'}')
        .text("version")
        .small_integer(1)
        .opcode(b's')
        .opcode(b'b')
        .text("temperature")
        .float(2.3510958125672174)
        .text("option_isolation")
        .opcode(0x89)
        .text("base")
        .memo_put(2)
        .memo_get(2)
        .opcode(b'u')
        .stop()
}

#[test]
fn should_read_a_dictionary_of_tensors_as_torch_save_writes_it() {
    let pickled_value = read_pickle(&head_like_pickle()).expect("a torch.save pickle should read");

    assert_eq!(
        pickled_value.member("temperature"),
        Some(&PickledValue::Float(2.3510958125672174))
    );
    assert_eq!(
        pickled_value.member("option_isolation"),
        Some(&PickledValue::Boolean(false))
    );
    assert_eq!(
        pickled_value.member("base"),
        Some(&PickledValue::Text("base".to_owned())),
        "a recalled memo entry holds the value it stored"
    );
    let head = pickled_value
        .member("head")
        .expect("the head should be present");
    assert_eq!(
        head.member("q.weight"),
        Some(&PickledValue::Tensor(PickledTensor {
            storage: PickledStorage {
                element_type: StorageElementType::Float32,
                storage_key: "0".to_owned(),
                element_count: 32,
            },
            storage_offset: 0,
            shape: vec![4, 8],
            strides: vec![8, 1],
        }))
    );
}

#[test]
fn should_read_integers_of_every_width_lists_and_tuples() {
    let pickle_bytes = PickleBuilder::default()
        .opcode(b']')
        .opcode(b'(')
        .small_integer(7)
        .medium_integer(1_000)
        .integer(-5)
        .opcode(b'N')
        .opcode(0x88)
        .opcode(b'e')
        .opcode(b')')
        .opcode(0x86)
        .stop();

    let pickled_value = read_pickle(&pickle_bytes).expect("plain values should read");

    assert_eq!(
        pickled_value,
        PickledValue::Tuple(vec![
            PickledValue::List(vec![
                PickledValue::Integer(7),
                PickledValue::Integer(1_000),
                PickledValue::Integer(-5),
                PickledValue::None,
                PickledValue::Boolean(true),
            ]),
            PickledValue::Tuple(Vec::new()),
        ])
    );
}

#[test]
fn should_refuse_a_callable_that_could_run_code() {
    let pickle_bytes = PickleBuilder::default()
        .global("os", "system")
        .text("echo unsafe")
        .opcode(0x85)
        .opcode(b'R')
        .stop();

    let read_outcome = read_pickle(&pickle_bytes);

    assert!(matches!(read_outcome, Err(reason) if reason.contains("os.system")));
}

#[test]
fn should_refuse_opcodes_outside_the_plain_data_set() {
    // INST (`i`) builds an instance of any class by name.
    let pickle_bytes = PickleBuilder::default().opcode(b'(').opcode(b'i').stop();

    assert!(matches!(read_pickle(&pickle_bytes), Err(reason) if reason.contains("0x69")));
}

#[test]
fn should_refuse_calling_an_allowed_name_in_a_way_torch_save_never_does() {
    let pickle_bytes = PickleBuilder::default()
        .global("collections", "OrderedDict")
        .text("unexpected argument")
        .opcode(0x85)
        .opcode(b'R')
        .stop();

    assert!(read_pickle(&pickle_bytes).is_err());
}

#[test]
fn should_refuse_a_shared_container_rather_than_copy_it() {
    let pickle_bytes = PickleBuilder::default()
        .opcode(b'}')
        .memo_put(0)
        .memo_get(0)
        .opcode(0x86)
        .stop();

    assert!(matches!(read_pickle(&pickle_bytes), Err(reason) if reason.contains("shares")));
}

#[test]
fn should_refuse_a_pickle_that_ends_early_or_leaves_values_behind() {
    let mut truncated = head_like_pickle();
    truncated.truncate(truncated.len() - 3);
    let leftover = PickleBuilder::default()
        .small_integer(1)
        .small_integer(2)
        .stop();

    assert!(read_pickle(&truncated).is_err());
    assert!(read_pickle(&leftover).is_err());
    assert!(read_pickle(&[]).is_err());
}

#[test]
fn should_refuse_recalling_a_memo_entry_that_was_never_stored() {
    let pickle_bytes = PickleBuilder::default().memo_get(9).stop();

    assert!(read_pickle(&pickle_bytes).is_err());
}
