typedef struct mlx_array_ mlx_array;
typedef enum mlx_fft_norm_ { MLX_FFT_NORM_NONE } mlx_fft_norm;
int mlx_fft_fft(mlx_array* res, mlx_fft_norm norm, const mlx_stream s);
