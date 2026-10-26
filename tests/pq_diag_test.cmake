# 同时检查退出状态与手算不变量，避免只匹配控制台字符串掩盖失败。
if(NOT DEFINED ANN_PQ_DIAG)
    message(FATAL_ERROR "ANN_PQ_DIAG executable is required")
endif()
execute_process(COMMAND "${ANN_PQ_DIAG}" --self-check
                RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE progress)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "PQ diagnostic failed: ${output} ${progress}")
endif()
foreach(expected
        "ivf_vs_exhaustive_adc_id_mismatches=0 distance_mismatches=0"
        "adc_vs_decoded_double_topk_position_mismatches=0"
        "max_relative_adc_distance_error=0"
        "strict_recall=1 decoded_double_recall=1"
        "reconstruction_mse_per_coordinate=0 residual_relative_squared_error=0"
        "lost_exact_neighbors=0")
    string(FIND "${output}" "${expected}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Missing hand-computed invariant: ${expected}")
    endif()
endforeach()
execute_process(COMMAND "${ANN_PQ_DIAG}"
                RESULT_VARIABLE status OUTPUT_QUIET ERROR_QUIET)
if(status EQUAL 0)
    message(FATAL_ERROR "Missing diagnostic arguments must fail")
endif()
