// The Float32 router-weighted sum of one output element of a sorted expert
// reduction, left in weighted_sum with output_index and token_index in scope.
// Every kernel that reduces sorted expert outputs starts with these
// statements, so their sums stay bit for bit equal: the accumulation order and
// the Float32 widening are what decide the rounded bits.

    auto output_index = thread_position_in_grid.x;
    auto token_index = output_index / output_dimension;
    auto output_dimension_index = output_index % output_dimension;
    float weighted_sum = 0.0f;
    for (int selected_expert_index = 0;
         selected_expert_index < selected_expert_count;
         ++selected_expert_index) {
        auto assignment_index =
            token_index * selected_expert_count + selected_expert_index;
        auto sorted_assignment_index = inverse_order[assignment_index];
        weighted_sum +=
            (float)sorted_outputs[sorted_assignment_index * output_dimension +
                                  output_dimension_index] *
            (float)scores[assignment_index];
    }
