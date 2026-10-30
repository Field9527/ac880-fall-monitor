# AC880 user LED1..7 and passive buzzer, manufacturer chapters2/19.
set_property IOSTANDARD LVCMOS33 [get_ports {home_leds[*] beep}]
set_property PACKAGE_PIN P22 [get_ports {home_leds[1]}]
set_property PACKAGE_PIN R18 [get_ports {home_leds[2]}]
set_property PACKAGE_PIN T18 [get_ports {home_leds[3]}]
set_property PACKAGE_PIN P20 [get_ports {home_leds[4]}]
set_property PACKAGE_PIN P21 [get_ports {home_leds[5]}]
set_property PACKAGE_PIN R20 [get_ports {home_leds[6]}]
set_property PACKAGE_PIN R21 [get_ports {home_leds[7]}]
set_property PACKAGE_PIN J22 [get_ports beep]
# Only the first stage of the automatic camera-ready indicator crosses clocks.
set_false_path -to [get_pins -hier -filter {NAME == "u_home_io/camera_sync_reg[0]/D"}]
