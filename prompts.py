# right
spark_prompt1 = "17/08/22 15:50:55 DEBUG BlockManager Putting block rdd_0_1 with replication took 0"

# right
spark_prompt2 = "17/08/22 15:51:02 INFO BlockManager Writing block rdd_1_3 to disk"

# right
windows_prompt1 = "2016-09-28 04:30:31, Info                  CBS    SQM: Failed to start upload with file pattern: C:\Windows\servicing\sqm\*_std.sqm, flags: 0x2 [HRESULT = 0x80004005 - E_FAIL]"

# right
windows_prompt2 = "2016-09-28 04:30:31, Info                  CSI    00000004 IAdvancedInstallerAwareStore_ResolvePendingTransactions (call 1) (flags = 00000004, progress = NULL, phase = 0, pdwDisposition = @0xb6fd90"

# right: log_printf("CBS    Warning: Unrecognized packageExtended attribute.")
windows_prompt3 = "2016-09-28 04:30:32, Info                  CBS    Warning: Unrecognized packageExtended attribute."

# right: log_printf("Failed to internally open package. [HRESULT = %d - CBS_E_INVALID_PACKAGE]")
windows_prompt4 = "2016-09-28 04:30:32, Info                  CBS    Failed to internally open package. [HRESULT = 0x800f0805 - CBS_E_INVALID_PACKAGE]"

# right: log_printf("Read out cached package applicability for package: %s, ApplicableState: %d, CurrentState: %d")
windows_prompt5 = "2016-09-28 04:30:33, Info                  CBS    Read out cached package applicability for package: Package_for_KB3121255~31bf3856ad364e35~amd64~~6.1.1.0, ApplicableState: 112, CurrentState:112"

# right: log_printf("SQM: Cleaning up report files older than %d days.", 10)
windows_prompt6 = "2016-09-28 04:30:31, Info                  CBS    SQM: Cleaning up report files older than 10 days."

# wrong: log_printf("Session: %s initialized by client %s", session_id, client_name)
# gt: Session: %s_%s initialized by client WindowsUpdateAgent.
windows_prompt7 = "2016-09-29 02:04:23, Info                  CBS    Session: 30546354_3192394775 initialized by client WindowsUpdateAgent."


# one_shot wrong: log_printf("%s NonStart: Checking to ensure startup processing was not required.")
# two_shot right: log_printf("NonStart: Checking to ensure startup processing was not required.")
windows_prompt8 = "2016-09-29 02:04:23, Info                  CBS    NonStart: Checking to ensure startup processing was not required."

# right: log_printf("Read out cached package applicability for package: %s, ApplicableState: %d, CurrentState: %d", package_name, applicable_state, current_state)
windows_prompt9 = "2016-09-29 02:04:23, Info                  CBS    Read out cached package applicability for package: Microsoft-Windows-Embedded-EmbeddedLockdown-Package-TopLevel~31bf3856ad364e35~amd64~~7.1.7601.16511, ApplicableState: 112, CurrentState:0"
