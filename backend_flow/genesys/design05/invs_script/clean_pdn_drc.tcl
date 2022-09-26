# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
set lx [dbGet top.fPlan.box_llx]
set ux [dbGet top.fPlan.box_urx]
set ly [dbGet top.fPlan.box_lly]
set uy [dbGet top.fPlan.box_ury]

set markers [dbGet top.markers]

set num_drc 0
foreach marker $markers {
    if {[dbGet $marker.type] != "Connectivity"} {
       incr num_drc
    }
}


while {$num_drc > 0} {
    puts "num_drc:   $num_drc"
    foreach marker $markers {
        if {[dbGet $marker.type] != "Connectivity"} {
            set layer [dbGet $marker.layer.name]
            set direction [dbGet $marker.layer.direction]
            set box_lx [dbGet $marker.box_llx]
            set box_ly [dbGet $marker.box_lly]
            set box_ux [dbGet $marker.box_urx]
            set box_uy [dbGet $marker.box_ury]
            if {$direction == "Horizontal"} {
                createRouteBlk -layer $layer -box  $lx  $box_ly $ux $box_uy    
            } else {
                createRouteBlk -layer $layer -box  $box_lx  $ly $box_ux $uy
            }
        }
    }

    source gf12_power_stripes.tcl

    verify_drc

    set markers [dbGet top.markers]

    set num_drc 0
    foreach marker $markers {
        if {[dbGet $marker.type] != "Connectivity"} {
            incr num_drc
        }
    }

    puts "num_drc:   $num_drc"
    if { $num_drc == 0 } {
      break
    } else {
      puts "num_drc:   $num_drc"
    }
}


deleteRouteBlk -all





