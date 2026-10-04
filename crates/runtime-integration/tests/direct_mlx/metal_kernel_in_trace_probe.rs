use std::cell::Cell;

use astronomical_runtime_integration::{
    MlxCompiledMultiOutputGraph, MlxDtype, MlxGraphBuilder, MlxMetalKernel, MlxMetalKernelOutput,
    MlxMetalKernelTemplateArgument, MlxStream, apply_metal_kernel_in_graph, array_from_vector, raw,
    set_graph_output,
};

use crate::common::runtime_test_support::{assert_f32_close, runtime};

const COMPILE_OPERATION: &str = "compile the metal-kernel-in-trace probe graph";
const ELEMENT_COUNT: usize = 3;

thread_local! {
    /// The probe kernel pinned for the duration of compile or apply, using
    /// the same raw-pointer context pattern as the verification-window
    /// geometry (the MLX-C closure ABI carries no capture context).
    static PROBE_KERNEL: Cell<*const MlxMetalKernel> = const { Cell::new(std::ptr::null()) };
}

fn with_pinned_kernel<R>(kernel: &MlxMetalKernel, trace_work: impl FnOnce() -> R) -> R {
    PROBE_KERNEL.with(|cell| {
        cell.set(kernel as *const MlxMetalKernel);
        let trace_result = trace_work();
        cell.set(std::ptr::null());
        trace_result
    })
}

/// Answers whether a fused Metal kernel can run inside a compiled-graph
/// trace: the gated-delta recurrence composes roughly ninety small ops per
/// layer, and a traceable fused kernel collapses them to one launch. The
/// probe applies a trivial scale-and-offset kernel inside the trace and
/// value-checks the replayed result. The compiled graph holds the kernel
/// primitive, so the kernel must outlive the graph.
#[test]
fn should_apply_a_fused_metal_kernel_inside_a_compiled_trace() {
    let runtime = runtime();
    let kernel = MlxMetalKernel::new(
        "probe_fused_scale_offset",
        &["input", "factor"],
        &["output"],
        // The generated kernel signature declares buffers named exactly after
        // the input and output names, so the source must reference those
        // identifiers rather than Metal shorthand like `in` and `out`. Scalar
        // (rank-zero) inputs are declared as references, not pointers, so
        // `factor` is used directly instead of `factor[0]`.
        &[
            "uint index = thread_position_in_grid.x;",
            "if (index >= uint(N)) { return; }",
            "float value = float(input[index]) * float(factor) + 1.0f;",
            "output[index] = static_cast<T>(value);",
        ]
        .join("\n"),
    )
    .expect("the probe kernel should compile");
    let compiled_graph = with_pinned_kernel(&kernel, || {
        MlxCompiledMultiOutputGraph::new(
            build_probe_graph as MlxGraphBuilder,
            COMPILE_OPERATION,
            false,
        )
        .expect("the probe graph should compile")
    });

    for replay_index in 0..2 {
        let input = runtime
            .array_from_f32(&[1.0, 2.0, 4.0], &[ELEMENT_COUNT as i32])
            .and_then(|float32_input| runtime.astype(&float32_input, MlxDtype::BFloat16))
            .expect("the input should build");
        let factor = runtime
            .array_from_f32(&[2.0], &[])
            .and_then(|float32_factor| runtime.astype(&float32_factor, MlxDtype::BFloat16))
            .expect("the factor should build");
        let outputs = with_pinned_kernel(&kernel, || {
            runtime.apply_compiled_multi_output_graph(&compiled_graph, &[&input, &factor])
        })
        .unwrap_or_else(|error| {
            panic!("replay {replay_index}: the compiled graph with a Metal kernel should apply: {error}")
        });
        assert_eq!(outputs.len(), 1);
        let float32_output = runtime
            .astype(&outputs[0], MlxDtype::Float32)
            .expect("the output should cast to float32");
        let values = float32_output
            .to_vec_f32()
            .expect("the output should read back");
        assert_f32_close(&values, &[3.0, 5.0, 9.0]);
    }
}

unsafe extern "C" fn build_probe_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> i32 {
    if output_vector.is_null() {
        return 1;
    }
    let kernel_pointer = PROBE_KERNEL.with(|cell| cell.get());
    if kernel_pointer.is_null() {
        return 1;
    }
    // SAFETY: The context pin guarantees the reference outlives this call.
    let kernel = unsafe { &*kernel_pointer };
    let input = match array_from_vector(input_vector, 0) {
        Ok(input) => input,
        Err(status) => return status,
    };
    let factor = match array_from_vector(input_vector, 1) {
        Ok(factor) => factor,
        Err(status) => return status,
    };
    let Ok(gpu_stream) = MlxStream::default_gpu() else {
        return 1;
    };
    let mut kernel_outputs = match apply_metal_kernel_in_graph(
        kernel,
        &[&input, &factor],
        &[MlxMetalKernelOutput::new(
            vec![ELEMENT_COUNT as i32],
            MlxDtype::BFloat16,
        )],
        [ELEMENT_COUNT as i32, 1, 1],
        [ELEMENT_COUNT as i32, 1, 1],
        &[
            MlxMetalKernelTemplateArgument::Integer {
                name: "N",
                integer_template_argument: ELEMENT_COUNT as i32,
            },
            MlxMetalKernelTemplateArgument::Dtype {
                name: "T",
                dtype: MlxDtype::BFloat16,
            },
        ],
        &gpu_stream,
    ) {
        Ok(kernel_outputs) => kernel_outputs,
        Err(status) => return status,
    };
    let Some(output) = kernel_outputs.pop() else {
        return 1;
    };
    // SAFETY: The output vector is unique and live for this publish.
    unsafe { set_graph_output(output_vector, &output) }
}
