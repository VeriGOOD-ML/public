`define WEIGHT_COUNT_MACRO(lanes,pe) (\
(lanes == 0 && pe == 0) ? 16'd0 : \
(lanes == 0 && pe == 1) ? 16'd0 : \
(lanes == 0 && pe == 2) ? 16'd0 : \
(lanes == 0 && pe == 3) ? 16'd0 : \
(lanes == 0 && pe == 4) ? 16'd0 : \
(lanes == 1 && pe == 0) ? 16'd4 : \
(lanes == 1 && pe == 1) ? 16'd4 : \
(lanes == 1 && pe == 2) ? 16'd4 : \
(lanes == 1 && pe == 3) ? 16'd3 : \
(lanes == 1 && pe == 4) ? 16'd15 : \
(lanes == 2 && pe == 0) ? 16'd4 : \
(lanes == 2 && pe == 1) ? 16'd4 : \
(lanes == 2 && pe == 2) ? 16'd4 : \
(lanes == 2 && pe == 3) ? 16'd3 : \
(lanes == 2 && pe == 4) ? 16'd15 : \
(lanes == 3 && pe == 0) ? 16'd4 : \
(lanes == 3 && pe == 1) ? 16'd4 : \
(lanes == 3 && pe == 2) ? 16'd4 : \
(lanes == 3 && pe == 3) ? 16'd3 : \
(lanes == 3 && pe == 4) ? 16'd15 : \
(lanes == 4 && pe == 0) ? 16'd4 : \
(lanes == 4 && pe == 1) ? 16'd4 : \
(lanes == 4 && pe == 2) ? 16'd4 : \
(lanes == 4 && pe == 3) ? 16'd3 : \
(lanes == 4 && pe == 4) ? 16'd15 : \
(lanes == 5 && pe == 0) ? 16'd4 : \
(lanes == 5 && pe == 1) ? 16'd4 : \
(lanes == 5 && pe == 2) ? 16'd3 : \
(lanes == 5 && pe == 3) ? 16'd3 : \
(lanes == 5 && pe == 4) ? 16'd14 : \
(lanes == 6 && pe == 0) ? 16'd4 : \
(lanes == 6 && pe == 1) ? 16'd4 : \
(lanes == 6 && pe == 2) ? 16'd3 : \
(lanes == 6 && pe == 3) ? 16'd3 : \
(lanes == 6 && pe == 4) ? 16'd14 : \
(lanes == 7 && pe == 0) ? 16'd4 : \
(lanes == 7 && pe == 1) ? 16'd4 : \
(lanes == 7 && pe == 2) ? 16'd3 : \
(lanes == 7 && pe == 3) ? 16'd3 : \
(lanes == 7 && pe == 4) ? 16'd14 : \
(lanes == 8 && pe == 0) ? 16'd0 : \
(lanes == 8 && pe == 1) ? 16'd0 : \
(lanes == 8 && pe == 2) ? 16'd0 : \
(lanes == 8 && pe == 3) ? 16'd0 : \
(lanes == 8 && pe == 4) ? 16'd0 : \
(lanes == 9 && pe == 0) ? 16'd4 : \
(lanes == 9 && pe == 1) ? 16'd4 : \
(lanes == 9 && pe == 2) ? 16'd3 : \
(lanes == 9 && pe == 3) ? 16'd3 : \
(lanes == 9 && pe == 4) ? 16'd14 : \
(lanes == 10 && pe == 0) ? 16'd4 : \
(lanes == 10 && pe == 1) ? 16'd4 : \
(lanes == 10 && pe == 2) ? 16'd3 : \
(lanes == 10 && pe == 3) ? 16'd3 : \
(lanes == 10 && pe == 4) ? 16'd14 : \
(lanes == 11 && pe == 0) ? 16'd4 : \
(lanes == 11 && pe == 1) ? 16'd4 : \
(lanes == 11 && pe == 2) ? 16'd3 : \
(lanes == 11 && pe == 3) ? 16'd3 : \
(lanes == 11 && pe == 4) ? 16'd14 : \
(lanes == 12 && pe == 0) ? 16'd4 : \
(lanes == 12 && pe == 1) ? 16'd4 : \
(lanes == 12 && pe == 2) ? 16'd3 : \
(lanes == 12 && pe == 3) ? 16'd3 : \
(lanes == 12 && pe == 4) ? 16'd14 : \
(lanes == 13 && pe == 0) ? 16'd4 : \
(lanes == 13 && pe == 1) ? 16'd4 : \
(lanes == 13 && pe == 2) ? 16'd3 : \
(lanes == 13 && pe == 3) ? 16'd3 : \
(lanes == 13 && pe == 4) ? 16'd14 : \
(lanes == 14 && pe == 0) ? 16'd4 : \
(lanes == 14 && pe == 1) ? 16'd4 : \
(lanes == 14 && pe == 2) ? 16'd3 : \
(lanes == 14 && pe == 3) ? 16'd3 : \
(lanes == 14 && pe == 4) ? 16'd14 : \
(lanes == 15 && pe == 0) ? 16'd4 : \
(lanes == 15 && pe == 1) ? 16'd4 : \
(lanes == 15 && pe == 2) ? 16'd3 : \
(lanes == 15 && pe == 3) ? 16'd3 : \
16'd14)