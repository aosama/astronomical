//! Generates the declaration of every Metal kernel this crate owns from its
//! `.metal` files (issue #2712): see `metal-kernel-declarations`.

fn main() {
    metal_kernel_declarations::run_kernel_declarations_build_script();
}
