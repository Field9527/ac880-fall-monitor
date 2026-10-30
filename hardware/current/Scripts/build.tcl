# 重建当前视频与外设工程，不连接或下载开发板。
set source_root [file dirname [file dirname [file normalize [info script]]]]
if {$argc < 1} {error "Usage: vivado -mode batch -source build.tcl -tclargs E:/build_directory"}
set build_dir [file normalize [lindex $argv 0]]
if {$::tcl_platform(platform) eq "windows" && ![string equal -nocase [string range $build_dir 0 1] "E:"]} {
    error "Build output must stay on E:"
}
if {[file exists [file join $build_dir ac880_video.xpr]]} {error "Use a new build directory"}
if {![string match "2025.2*" [version -short]]} {error "Vivado 2025.2 is required"}
file mkdir $build_dir
create_project ac880_video $build_dir -part xc7z020clg484-2
set_property target_language Verilog [current_project]
config_ip_cache -use_cache_location E:/Zynq_cache/vivado_ip
set_param general.maxThreads 4
add_files -norecurse [glob [file join $source_root Sources RTL *.v]]
add_files -fileset constrs_1 -norecurse [glob [file join $source_root Sources Constraints *.xdc]]
source [file join $source_root Sources BD system.tcl]
set bd [get_files -all *system.bd]
if {[llength $bd] != 1} {error "Expected one system block design"}
open_bd_design $bd
validate_bd_design
if {[llength [get_bd_cells -quiet {cnn_accel_0 pointwise_0 pl_memory}]]} {error "Obsolete inference endpoints found"}
foreach name {processing_system7_0 axi_vdma_0 frame_copy smartconnect_0} {
    if {[llength [get_bd_cells -quiet $name]] != 1} {error "Missing retained IP: $name"}
}
set_property synth_checkpoint_mode None $bd
add_files -norecurse [make_wrapper -files $bd -top]
source [file join $source_root Scripts project_settings.tcl]
update_compile_order -fileset sources_1
report_ip_status -file [file join $build_dir ip_status.rpt]
puts "AC880_SOURCE_REBUILD_PASS top=[get_property TOP [get_filesets sources_1]]"
close_project
