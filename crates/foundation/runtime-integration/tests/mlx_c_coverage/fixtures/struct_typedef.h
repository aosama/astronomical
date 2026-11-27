typedef struct mlx_stream_ {
void* ctx;
} mlx_stream;
int mlx_stream_to_string(mlx_string** res, const mlx_stream s);
