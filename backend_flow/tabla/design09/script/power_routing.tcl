# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
setAddStripeMode -route_over_rows_only true
setAddStripeMode -stacked_via_bottom_layer M1 -stacked_via_top_layer C5
addStripe -nets { VSS VDD } \
          -layer C5 \
          -direction vertical \
          -ybottom_offset 9.6 \
          -xleft_offset 9.6 \
          -width 0.308 \
          -spacing 0.332 \
          -set_to_set_distance 9.6 \
          -extend_to all_domains


setAddStripeMode -reset
setAddStripeMode -orthogonal_only false -ignore_DRC false
setViaGenMode -ignore_DRC false

setAddStripeMode -stacked_via_bottom_layer C5 -stacked_via_top_layer K3
addStripe -nets { VSS VDD }  \
          -layer K3 \
          -direction horizontal \
          -ybottom_offset 6.4 \
          -xleft_offset 6.4 \
          -width 0.308 \
          -spacing 12.492 \
          -set_to_set_distance 25.6 \
          -extend_to all_domains
 
setAddStripeMode -stacked_via_bottom_layer K3 -stacked_via_top_layer K4
addStripe -nets { VSS VDD } \
          -layer K4 \
          -direction vertical \
          -ybottom_offset 6.4 \
          -xleft_offset 6.4 \
          -width 0.308 \
          -spacing 6.092 \
          -set_to_set_distance 12.8 \
          -extend_to all_domains
