uint index = thread_position_in_grid.x;
if (index >= uint(N)) { return; }
float value = float(input[index]) * float(factor) + 1.0f;
output[index] = static_cast<T>(value);