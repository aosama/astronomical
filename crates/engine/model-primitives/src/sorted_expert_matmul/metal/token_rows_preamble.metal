// The sorted rows are read through the sort: X holds one row per token, and
// sorted row r is token order[r] / TOPK's row, so no sorted copy of the
// token rows is ever made. Only the staged layouts read through it.
#define SOURCE_ROW(sorted_row) long(uint(order[sorted_row]) / uint(TOPK))
