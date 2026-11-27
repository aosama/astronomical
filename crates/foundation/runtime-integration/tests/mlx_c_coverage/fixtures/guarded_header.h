#ifndef MLX_GUARD_H
#define MLX_GUARD_H
#ifdef __cplusplus
extern "C" {
#endif
int mlx_version(mlx_string* str_);
#ifdef HAS_BFLOAT16
int mlx_array_item_bfloat16(uint16_t* res, const mlx_array arr);
#endif
#ifdef __cplusplus
}
#endif
#endif
