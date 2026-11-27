//! The declaration of one registered kernel, as its owning crate's build
//! script generates it from the kernel's `.metal` file.

/// What a launcher hands MLX to declare one kernel: the registered name, the
/// input and output names in MLX's buffer order, and the header and source
/// texts, each the concatenation of its files in the order the signature
/// block lists them. Generated, never written by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalKernelSource {
    /// The file stem, which is the registered kernel name.
    pub name: &'static str,
    pub input_names: &'static [&'static str],
    pub output_names: &'static [&'static str],
    /// The template parameters the launcher must pass, in MLX's order.
    pub template_parameter_names: &'static [&'static str],
    /// Values the launcher bakes into the source as `#define` lines, for
    /// numbers MLX's template arguments cannot carry (a float scale).
    pub define_names: &'static [&'static str],
    /// Whether MLX declares the outputs `atomic<T>`, for kernels whose
    /// threads add into the same output element.
    pub has_atomic_outputs: bool,
    pub header_parts: &'static [&'static str],
    pub source_parts: &'static [&'static str],
}

/// One `#define` value a launcher supplies, written as Metal text
/// (`1.000000000e-06f`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalKernelDefine {
    pub name: &'static str,
    pub value: String,
}

/// The launcher's defines do not match the kernel's declared ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalKernelDefineError {
    pub kernel_name: &'static str,
    pub description: String,
}

impl std::fmt::Display for MetalKernelDefineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "kernel {}: {}",
            self.kernel_name, self.description
        )
    }
}

impl std::error::Error for MetalKernelDefineError {}

impl MetalKernelSource {
    /// The header MLX places ahead of the generated signature; empty when the
    /// kernel names no header.
    pub fn header_text(&self) -> String {
        self.header_parts.concat()
    }

    /// The body MLX places inside the generated function: one `#define` line
    /// per declared define, in declaration order, then the source files.
    ///
    /// # Errors
    /// Fails when a declared define is missing, supplied twice, or when a
    /// define the kernel does not declare is supplied.
    pub fn source_text(
        &self,
        defines: &[MetalKernelDefine],
    ) -> Result<String, MetalKernelDefineError> {
        self.check_defines(defines)?;
        let mut source_text = String::new();
        for define_name in self.define_names {
            for define in defines {
                if define.name == *define_name {
                    source_text.push_str(&format!("#define {} {}\n", define.name, define.value));
                }
            }
        }
        for source_part in self.source_parts {
            source_text.push_str(source_part);
        }
        Ok(source_text)
    }

    fn check_defines(&self, defines: &[MetalKernelDefine]) -> Result<(), MetalKernelDefineError> {
        for define in defines {
            if !self.define_names.contains(&define.name) {
                return Err(self.define_error(format!("{} is not a declared define", define.name)));
            }
        }
        for define_name in self.define_names {
            let mut supplied_count = 0;
            for define in defines {
                if define.name == *define_name {
                    supplied_count += 1;
                }
            }
            if supplied_count != 1 {
                return Err(self.define_error(format!(
                    "define {define_name} was supplied {supplied_count} times; it needs exactly one value"
                )));
            }
        }
        Ok(())
    }

    fn define_error(&self, description: String) -> MetalKernelDefineError {
        MetalKernelDefineError {
            kernel_name: self.name,
            description,
        }
    }
}
