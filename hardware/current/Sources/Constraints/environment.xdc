# P7 pins 29/30 supply 3.3V/GND. Adjacent P7-31/32 carry both signals.
set_property PACKAGE_PIN V8 [get_ports dht11_data]
set_property PACKAGE_PIN W8 [get_ports light_do]
set_property IOSTANDARD LVCMOS33 [get_ports {dht11_data light_do}]
set_property PULLUP true [get_ports {dht11_data light_do}]
set_property DRIVE 4 [get_ports dht11_data]
set_property SLEW SLOW [get_ports dht11_data]
# External asynchronous inputs terminate at the first synchronizer stage only.
set_false_path -from [get_ports dht11_data] -to [get_pins -hier -filter {NAME == "u_environment/dht_sync_reg[0]/D"}]
set_false_path -from [get_ports light_do] -to [get_pins -hier -filter {NAME == "u_environment/light_sync_reg[0]/D"}]
# The outbound open-drain pulse is an asynchronous, microsecond bus protocol.
set_false_path -to [get_ports dht11_data]
