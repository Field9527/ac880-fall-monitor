# Independent PL-acceleration build: constrain only asynchronous reset assertion.
# fall_top rst_sync_cam is a four-stage asynchronous-assert/synchronous-release
# chain. Its PRE pins are asynchronous to the OV5640 pixel clock. D/Q paths
# between the stages and from the final stage to camera-domain users remain timed.
set camera_reset_stages [get_cells -hier -filter {NAME =~ *rst_sync_cam_reg* && IS_SEQUENTIAL == 1}]
set camera_reset_assert [get_pins -of_objects $camera_reset_stages -filter {REF_PIN_NAME == PRE}]
set_property ASYNC_REG TRUE $camera_reset_stages
set_false_path -to $camera_reset_assert
