# 同时检查退出状态与手算不变量，避免只匹配控制台字符串掩盖失败。
if(NOT DEFINED ANN_PQ_DIAG)
    message(FATAL_ERROR "ANN_PQ_DIAG executable is required")
endif()
foreach(iterations IN ITEMS default 25)
    set(arguments --self-check)
    set(expected_iterations 5)
    if(NOT iterations STREQUAL "default")
        list(APPEND arguments "${iterations}")
        set(expected_iterations "${iterations}")
    endif()
    execute_process(COMMAND "${ANN_PQ_DIAG}" ${arguments}
                    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE progress)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "PQ diagnostic failed: ${output} ${progress}")
    endif()
    foreach(expected
            "coarse_iters=${expected_iterations} pq_iters=25"
            "ivf_vs_exhaustive_adc_id_mismatches=0 distance_mismatches=0"
            "adc_vs_decoded_double_topk_position_mismatches=0"
            "max_relative_adc_distance_error=0"
            "strict_recall=1 decoded_double_recall=1"
            "reconstruction_mse_per_coordinate=0 residual_relative_squared_error=0"
            "coarse_residual_mse_per_coordinate=1 index_capacity_bytes="
            "lost_exact_neighbors=0")
        string(FIND "${output}" "${expected}" position)
        if(position EQUAL -1)
            message(FATAL_ERROR "Missing hand-computed invariant: ${expected}")
        endif()
    endforeach()
endforeach()
# 参数解析必须先于数据加载，检查无效轮数的明确报错。
foreach(invalid IN ITEMS 0 -1 1x 18446744073709551616)
    execute_process(COMMAND "${ANN_PQ_DIAG}" missing-data 1 2 "${invalid}"
                    RESULT_VARIABLE status OUTPUT_QUIET ERROR_VARIABLE diagnostic)
    if(status EQUAL 0 OR NOT diagnostic MATCHES "Expected a positive integer")
        message(FATAL_ERROR "Invalid iteration count was not rejected: ${invalid}")
    endif()
endforeach()
execute_process(COMMAND "${ANN_PQ_DIAG}"
                RESULT_VARIABLE status OUTPUT_QUIET ERROR_QUIET)
if(status EQUAL 0)
    message(FATAL_ERROR "Missing diagnostic arguments must fail")
endif()
