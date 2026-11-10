//! A pickle reader that interprets data and never runs code.
//!
//! `torch.save` writes Python's pickle format, which can name any Python
//! function and call it while loading: unpickling a file you did not write is
//! running code you did not write. A converter needs only the data inside,
//! so this reader interprets the opcodes that describe plain values (numbers,
//! text, lists, tuples, dictionaries) and allows exactly three named callables,
//! the ones a dictionary of tensors needs: `collections.OrderedDict`,
//! `torch._utils._rebuild_tensor_v2`, and the storage types. Calling one only
//! builds a description here; any other name, or any opcode outside this
//! set, makes the whole file refused.

use std::collections::HashMap;

/// Every opcode a file may hold; a larger file is not a decision head.
const MAXIMUM_OPCODE_COUNT: usize = 1_000_000;

/// One value read from a pickle.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PickledValue {
    None,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    Text(String),
    List(Vec<PickledValue>),
    Tuple(Vec<PickledValue>),
    Dictionary(Vec<(PickledValue, PickledValue)>),
    /// A reference to one of the allowed callables, not yet called.
    Callable(AllowedCallable),
    /// A tensor storage: which archive entry holds its bytes.
    Storage(PickledStorage),
    Tensor(PickledTensor),
}

/// The only callables a file may name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AllowedCallable {
    OrderedDictionary,
    RebuildTensor,
    StorageType(StorageElementType),
}

/// The element types of the storages a decision head uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StorageElementType {
    Float32,
    Float16,
    BFloat16,
}

impl StorageElementType {
    pub(crate) const fn bytes_per_element(self) -> u64 {
        match self {
            Self::Float32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PickledStorage {
    pub(crate) element_type: StorageElementType,
    /// The name of the archive entry under `data/` holding the bytes.
    pub(crate) storage_key: String,
    pub(crate) element_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PickledTensor {
    pub(crate) storage: PickledStorage,
    /// Where the tensor starts in its storage, in elements.
    pub(crate) storage_offset: u64,
    pub(crate) shape: Vec<u64>,
    pub(crate) strides: Vec<u64>,
}

impl PickledValue {
    /// The value stored under the text key `key`, when this is a dictionary.
    pub(crate) fn member(&self, key: &str) -> Option<&PickledValue> {
        let Self::Dictionary(members) = self else {
            return None;
        };
        members
            .iter()
            .find(|(member_key, _)| matches!(member_key, Self::Text(text) if text == key))
            .map(|(_, member_value)| member_value)
    }
}

enum StackEntry {
    Value(PickledValue),
    Mark,
}

struct Unpickler<'pickle> {
    pickle_bytes: &'pickle [u8],
    position: usize,
    stack: Vec<StackEntry>,
    memo: HashMap<u32, PickledValue>,
}

/// Reads one pickle and returns the value it describes.
pub(crate) fn read_pickle(pickle_bytes: &[u8]) -> Result<PickledValue, String> {
    let mut unpickler = Unpickler {
        pickle_bytes,
        position: 0,
        stack: Vec::new(),
        memo: HashMap::new(),
    };
    for _ in 0..MAXIMUM_OPCODE_COUNT {
        let opcode = unpickler.read_bytes(1)?[0];
        if opcode == b'.' {
            return unpickler.finish();
        }
        unpickler.apply_opcode(opcode)?;
    }
    Err(format!(
        "the pickle holds more than {MAXIMUM_OPCODE_COUNT} opcodes"
    ))
}

impl Unpickler<'_> {
    fn apply_opcode(&mut self, opcode: u8) -> Result<(), String> {
        match opcode {
            0x80 | b'X' | b'N' | 0x88 | 0x89 => self.push_literal(opcode),
            b'J' | b'K' | b'M' | b'G' => self.push_number(opcode),
            b'}' | b']' | b')' | b'(' | b't' | 0x85 | 0x86 | 0x87 => self.build_container(opcode),
            b's' | b'u' | b'a' | b'e' => self.fill_container(opcode),
            b'q' | b'r' | b'h' | b'j' => self.use_memo(opcode),
            b'c' | b'R' | b'Q' | b'b' => self.apply_object_opcode(opcode),
            unsupported_opcode => Err(format!(
                "the pickle uses opcode {unsupported_opcode:#04x}, which only code-running files need"
            )),
        }
    }

    fn push_literal(&mut self, opcode: u8) -> Result<(), String> {
        let literal_value = match opcode {
            0x80 => {
                self.read_bytes(1)?;
                return Ok(());
            }
            b'X' => {
                let text_length = self.read_u32()? as usize;
                let text_bytes = self.read_bytes(text_length)?;
                PickledValue::Text(
                    String::from_utf8(text_bytes.to_vec())
                        .map_err(|_| "the pickle holds text that is not UTF-8".to_owned())?,
                )
            }
            b'N' => PickledValue::None,
            other_opcode => PickledValue::Boolean(other_opcode == 0x88),
        };
        self.stack.push(StackEntry::Value(literal_value));
        Ok(())
    }

    fn push_number(&mut self, opcode: u8) -> Result<(), String> {
        let number_value = match opcode {
            b'J' => PickledValue::Integer(i64::from(self.read_u32()? as i32)),
            b'K' => PickledValue::Integer(i64::from(self.read_bytes(1)?[0])),
            b'M' => {
                let integer_bytes = self.read_bytes(2)?;
                PickledValue::Integer(i64::from(u16::from_le_bytes([
                    integer_bytes[0],
                    integer_bytes[1],
                ])))
            }
            _ => {
                let float_bytes: [u8; 8] = self
                    .read_bytes(8)?
                    .try_into()
                    .map_err(|_| "the pickle cut a float short".to_owned())?;
                PickledValue::Float(f64::from_be_bytes(float_bytes))
            }
        };
        self.stack.push(StackEntry::Value(number_value));
        Ok(())
    }

    fn build_container(&mut self, opcode: u8) -> Result<(), String> {
        let container = match opcode {
            b'}' => PickledValue::Dictionary(Vec::new()),
            b']' => PickledValue::List(Vec::new()),
            b')' => PickledValue::Tuple(Vec::new()),
            b'(' => {
                self.stack.push(StackEntry::Mark);
                return Ok(());
            }
            b't' => PickledValue::Tuple(self.pop_to_mark()?),
            tuple_opcode => {
                let element_count = usize::from(tuple_opcode - 0x84);
                let mut elements = Vec::with_capacity(element_count);
                for _ in 0..element_count {
                    elements.push(self.pop_value()?);
                }
                elements.reverse();
                PickledValue::Tuple(elements)
            }
        };
        self.stack.push(StackEntry::Value(container));
        Ok(())
    }

    fn fill_container(&mut self, opcode: u8) -> Result<(), String> {
        let added_values = match opcode {
            b's' | b'a' => {
                let item_count = if opcode == b's' { 2 } else { 1 };
                let mut items = Vec::with_capacity(item_count);
                for _ in 0..item_count {
                    items.push(self.pop_value()?);
                }
                items.reverse();
                items
            }
            _ => self.pop_to_mark()?,
        };
        let mut container = self.pop_value()?;
        match (&mut container, opcode) {
            (PickledValue::Dictionary(members), b's' | b'u') if added_values.len() % 2 == 0 => {
                let mut added_values = added_values.into_iter();
                while let (Some(key), Some(value)) = (added_values.next(), added_values.next()) {
                    members.push((key, value));
                }
            }
            (PickledValue::List(elements), b'a' | b'e') => elements.extend(added_values),
            _ => return Err("the pickle adds items to something that cannot hold them".to_owned()),
        }
        self.stack.push(StackEntry::Value(container));
        Ok(())
    }

    fn use_memo(&mut self, opcode: u8) -> Result<(), String> {
        let memo_key = if matches!(opcode, b'q' | b'h') {
            u32::from(self.read_bytes(1)?[0])
        } else {
            self.read_u32()?
        };
        if matches!(opcode, b'q' | b'r') {
            let Some(StackEntry::Value(top_value)) = self.stack.last() else {
                return Err("the pickle stores an empty stack in its memo".to_owned());
            };
            self.memo.insert(memo_key, top_value.clone());
            return Ok(());
        }
        let Some(remembered_value) = self.memo.get(&memo_key) else {
            return Err(format!(
                "the pickle recalls memo entry {memo_key}, which it never stored"
            ));
        };
        // A recalled container would be a copy of what the memo saw, not the
        // shared object pickle means, so a file that relies on sharing one is
        // refused rather than read wrongly.
        if matches!(
            remembered_value,
            PickledValue::List(_) | PickledValue::Dictionary(_)
        ) {
            return Err(
                "the pickle shares a list or dictionary, which this reader does not support"
                    .to_owned(),
            );
        }
        self.stack.push(StackEntry::Value(remembered_value.clone()));
        Ok(())
    }

    fn apply_object_opcode(&mut self, opcode: u8) -> Result<(), String> {
        let built_value = match opcode {
            b'c' => PickledValue::Callable(self.read_allowed_global()?),
            b'R' => {
                let arguments = self.pop_value()?;
                let callable = self.pop_value()?;
                call_allowed(callable, arguments)?
            }
            b'Q' => storage_from_persistent_id(self.pop_value()?)?,
            _ => {
                // BUILD sets an object's attributes. The only object here is an
                // OrderedDict, whose attribute is PyTorch's per-module version
                // record, which a converter does not need.
                self.pop_value()?;
                let Some(StackEntry::Value(PickledValue::Dictionary(_))) = self.stack.last() else {
                    return Err("the pickle builds an object this reader does not allow".to_owned());
                };
                return Ok(());
            }
        };
        self.stack.push(StackEntry::Value(built_value));
        Ok(())
    }

    fn read_allowed_global(&mut self) -> Result<AllowedCallable, String> {
        let module_name = self.read_line()?;
        let attribute_name = self.read_line()?;
        let allowed_callable = match (module_name.as_str(), attribute_name.as_str()) {
            ("collections", "OrderedDict") => AllowedCallable::OrderedDictionary,
            ("torch._utils", "_rebuild_tensor_v2") => AllowedCallable::RebuildTensor,
            ("torch", "FloatStorage") => AllowedCallable::StorageType(StorageElementType::Float32),
            ("torch", "HalfStorage") => AllowedCallable::StorageType(StorageElementType::Float16),
            ("torch", "BFloat16Storage") => {
                AllowedCallable::StorageType(StorageElementType::BFloat16)
            }
            _ => {
                return Err(format!(
                    "the pickle names {module_name}.{attribute_name}, which this reader does not allow"
                ));
            }
        };
        Ok(allowed_callable)
    }

    fn finish(mut self) -> Result<PickledValue, String> {
        let top_value = self.pop_value()?;
        if !self.stack.is_empty() {
            return Err("the pickle leaves values behind when it stops".to_owned());
        }
        Ok(top_value)
    }

    fn pop_value(&mut self) -> Result<PickledValue, String> {
        match self.stack.pop() {
            Some(StackEntry::Value(value)) => Ok(value),
            _ => Err("the pickle reads a value that is not there".to_owned()),
        }
    }

    fn pop_to_mark(&mut self) -> Result<Vec<PickledValue>, String> {
        let mut values = Vec::new();
        loop {
            match self.stack.pop() {
                Some(StackEntry::Value(value)) => values.push(value),
                Some(StackEntry::Mark) => {
                    values.reverse();
                    return Ok(values);
                }
                None => return Err("the pickle closes a group it never opened".to_owned()),
            }
        }
    }

    fn read_bytes(&mut self, byte_count: usize) -> Result<&[u8], String> {
        let read_end = self.position.saturating_add(byte_count);
        let Some(read_bytes) = self.pickle_bytes.get(self.position..read_end) else {
            return Err("the pickle ends in the middle of an opcode".to_owned());
        };
        self.position = read_end;
        Ok(read_bytes)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let integer_bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([
            integer_bytes[0],
            integer_bytes[1],
            integer_bytes[2],
            integer_bytes[3],
        ]))
    }

    fn read_line(&mut self) -> Result<String, String> {
        let remaining_bytes = &self.pickle_bytes[self.position.min(self.pickle_bytes.len())..];
        let Some(line_length) = remaining_bytes.iter().position(|byte| *byte == b'\n') else {
            return Err("the pickle ends in the middle of a name".to_owned());
        };
        let line_bytes = self.read_bytes(line_length + 1)?;
        String::from_utf8(line_bytes[..line_length].to_vec())
            .map_err(|_| "the pickle holds a name that is not UTF-8".to_owned())
    }
}

/// What calling an allowed callable with `arguments` describes.
fn call_allowed(callable: PickledValue, arguments: PickledValue) -> Result<PickledValue, String> {
    let PickledValue::Tuple(arguments) = arguments else {
        return Err("the pickle calls something with arguments that are not a tuple".to_owned());
    };
    match callable {
        PickledValue::Callable(AllowedCallable::OrderedDictionary) if arguments.is_empty() => {
            Ok(PickledValue::Dictionary(Vec::new()))
        }
        PickledValue::Callable(AllowedCallable::RebuildTensor) => rebuilt_tensor(arguments),
        _ => Err("the pickle calls something this reader does not allow".to_owned()),
    }
}

/// `_rebuild_tensor_v2(storage, offset, shape, strides, requires_grad, hooks, ...)`.
fn rebuilt_tensor(arguments: Vec<PickledValue>) -> Result<PickledValue, String> {
    let malformed = || "the pickle rebuilds a tensor from malformed arguments".to_owned();
    let mut arguments = arguments.into_iter();
    let Some(PickledValue::Storage(storage)) = arguments.next() else {
        return Err(malformed());
    };
    let Some(PickledValue::Integer(storage_offset)) = arguments.next() else {
        return Err(malformed());
    };
    let shape = whole_number_tuple(arguments.next()).ok_or_else(malformed)?;
    let strides = whole_number_tuple(arguments.next()).ok_or_else(malformed)?;
    if shape.len() != strides.len() || storage_offset < 0 {
        return Err(malformed());
    }
    Ok(PickledValue::Tensor(PickledTensor {
        storage,
        storage_offset: storage_offset as u64,
        shape,
        strides,
    }))
}

fn whole_number_tuple(value: Option<PickledValue>) -> Option<Vec<u64>> {
    let Some(PickledValue::Tuple(elements)) = value else {
        return None;
    };
    let mut whole_numbers = Vec::with_capacity(elements.len());
    for element in elements {
        let PickledValue::Integer(integer) = element else {
            return None;
        };
        whole_numbers.push(u64::try_from(integer).ok()?);
    }
    Some(whole_numbers)
}

/// `('storage', StorageType, key, location, element_count)`, PyTorch's
/// persistent id for one tensor storage held in the archive.
fn storage_from_persistent_id(persistent_id: PickledValue) -> Result<PickledValue, String> {
    let malformed =
        || "the pickle names a tensor storage in a form this reader does not know".to_owned();
    let PickledValue::Tuple(fields) = persistent_id else {
        return Err(malformed());
    };
    let [
        PickledValue::Text(kind),
        PickledValue::Callable(AllowedCallable::StorageType(element_type)),
        PickledValue::Text(storage_key),
        PickledValue::Text(_location),
        PickledValue::Integer(element_count),
    ] = fields.as_slice()
    else {
        return Err(malformed());
    };
    if kind != "storage" || *element_count < 0 {
        return Err(malformed());
    }
    Ok(PickledValue::Storage(PickledStorage {
        element_type: *element_type,
        storage_key: storage_key.clone(),
        element_count: *element_count as u64,
    }))
}

#[cfg(test)]
#[path = "torch_pickle_tests.rs"]
mod tests;
