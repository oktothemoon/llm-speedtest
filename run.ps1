#Requires -Version 5.1
# llm-speedtest 一键启动脚本（Windows PowerShell）
# 无需预先安装 Git；自动检测 Python 3.9+（避开微软商店空壳），缺失时提供中文指引或 winget 自动安装。
# 用法:
#   irm https://raw.githubusercontent.com/oktothemoon/llm-speedtest/main/run.ps1 | iex
# 可选环境变量:
#   $env:LLM_SPEEDTEST_BASE_URL  自定义下载源（默认 GitHub raw）
#   $env:LLM_SPEEDTEST_DIR       本地安装目录（默认 $HOME\llm-speedtest）

function Invoke-LlmSpeedtest {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$ForwardArgs
    )

    $ErrorActionPreference = "Stop"

    try {
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    } catch {}

    try {
        [Net.ServicePointManager]::SecurityProtocol = `
            [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch {}

    $BaseUrl = if ($env:LLM_SPEEDTEST_BASE_URL) {
        $env:LLM_SPEEDTEST_BASE_URL.TrimEnd('/')
    } else {
        "https://raw.githubusercontent.com/oktothemoon/llm-speedtest/main"
    }

    $InstallDir = if ($env:LLM_SPEEDTEST_DIR) {
        $env:LLM_SPEEDTEST_DIR
    } else {
        Join-Path $HOME "llm-speedtest"
    }

    function Write-Info([string]$Msg) { Write-Host "[llm-speedtest] $Msg" -ForegroundColor Cyan }
    function Write-Ok([string]$Msg)   { Write-Host "[llm-speedtest] $Msg" -ForegroundColor Green }
    function Write-Warn([string]$Msg) { Write-Host "[llm-speedtest] $Msg" -ForegroundColor Yellow }
    function Write-Err([string]$Msg)  { Write-Host "[llm-speedtest] $Msg" -ForegroundColor Red }

    $script:FoundOldPython = $null

    function Test-PythonCandidate {
        param(
            [string]$Exe,
            [string[]]$Prefix = @()
        )
        if (-not $Exe) { return $null }
        if ($Exe -like "*\WindowsApps\*") { return $null }
        if (-not (Test-Path $Exe -ErrorAction SilentlyContinue)) { return $null }

        try {
            $verOut = & $Exe @Prefix -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $verOut) { return $null }
            $verStr = ($verOut | Select-Object -First 1).ToString().Trim()
            & $Exe @Prefix -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Exe = $Exe; Prefix = $Prefix; Version = $verStr }
            }
            if (-not $script:FoundOldPython) {
                $script:FoundOldPython = "$verStr ($Exe)"
            }
        } catch {}
        return $null
    }

    function Find-Python {
        $script:FoundOldPython = $null

        # 1. PATH 中的 py / python / python3（跳过 WindowsApps 微软商店空壳）
        $cmdPy = Get-Command py -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cmdPy -and $cmdPy.Source) {
            $res = Test-PythonCandidate -Exe $cmdPy.Source -Prefix @("-3")
            if ($res) { return $res }
        }

        foreach ($name in @("python", "python3")) {
            $cmd = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($cmd -and $cmd.Source) {
                $res = Test-PythonCandidate -Exe $cmd.Source
                if ($res) { return $res }
            }
        }

        # 2. 常见官方安装路径（处理刚装完还没刷新 PATH 的情况）
        $searchRoots = @(
            (Join-Path $env:LOCALAPPDATA "Programs\Python"),
            $env:ProgramFiles,
            ${env:ProgramFiles(x86)}
        ) | Where-Object { $_ -and (Test-Path $_) }

        foreach ($root in $searchRoots) {
            $dirs = Get-ChildItem -Path $root -Directory -Filter "Python3*" -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending
            foreach ($d in $dirs) {
                $exe = Join-Path $d.FullName "python.exe"
                $res = Test-PythonCandidate -Exe $exe
                if ($res) { return $res }
            }
        }

        return $null
    }

    function Refresh-SessionPath {
        $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $parts = @($machinePath, $userPath) | Where-Object { $_ }
        if ($parts.Count -gt 0) {
            $env:Path = $parts -join ";"
        }
    }

    function Show-ManualPythonGuide {
        Write-Host ""
        Write-Info "请按以下步骤安装 Python 3.9+（约 1 分钟）："
        Write-Host "  1. 打开官方下载页：https://www.python.org/downloads/windows/"
        Write-Host "  2. 运行安装包时，务必勾选底部的 [Add python.exe to PATH]，再点 Install Now" -ForegroundColor Yellow
        Write-Host "  3. 安装完成后，重新打开一个 PowerShell 窗口，再次粘贴运行本命令即可。"
        Write-Host ""
    }

    function Download-FileSafe {
        param(
            [string]$Url,
            [string]$Dest
        )
        $tmp = "$Dest.tmp.$PID"
        try {
            $iwrParams = @{
                Uri     = $Url
                OutFile = $tmp
                TimeoutSec = 60
            }
            if ($PSVersionTable.PSVersion.Major -lt 6) {
                $iwrParams["UseBasicParsing"] = $true
            }
            Invoke-WebRequest @iwrParams
            if ((Test-Path $tmp) -and ((Get-Item $tmp).Length -gt 0)) {
                Move-Item -Path $tmp -Destination $Dest -Force
                return $true
            }
        } catch {}
        if (Test-Path $tmp) {
            Remove-Item -Path $tmp -Force -ErrorAction SilentlyContinue
        }
        return $false
    }

    # --- 第 1 步：检测或引导安装 Python 3.9+ ---
    $py = Find-Python
    if (-not $py) {
        Write-Host ""
        if ($script:FoundOldPython) {
            Write-Warn "检测到当前 Python 版本过低：$($script:FoundOldPython)（本工具需要 Python 3.9 及以上版本）。"
        } else {
            Write-Warn "未检测到可用的 Python 3.9+ 环境（本工具基于纯 Python 标准库，无需额外第三方依赖）。"
        }

        $winget = Get-Command winget -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($winget) {
            Write-Host ""
            Write-Info "检测到系统自带 winget，可一键安装官方 Python 3.12（用户级安装，不影响系统）："
            Write-Host "  winget install -e --id Python.Python.3.12 --scope user"
            Write-Host ""
            $ans = ""
            try {
                $ans = Read-Host "是否现在使用 winget 自动安装 Python 3.12？[Y/n]"
            } catch {
                $ans = "n"
            }
            if ([string]::IsNullOrWhiteSpace($ans) -or $ans.Trim().ToLower() -in @("y", "yes")) {
                Write-Info "正在通过 winget 安装 Python 3.12，请稍候..."
                try {
                    & winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
                } catch {}
                Refresh-SessionPath
                $py = Find-Python
                if ($py) {
                    Write-Ok "Python $($py.Version) 安装成功：$($py.Exe)"
                } else {
                    Write-Warn "winget 安装未生效或需要新终端才能生效。"
                    Show-ManualPythonGuide
                    return
                }
            } else {
                Show-ManualPythonGuide
                return
            }
        } else {
            Show-ManualPythonGuide
            return
        }
    }

    # --- 第 2 步：获取 / 更新 speedtest.py 和 ui.py（无需 Git） ---
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }

    $speedtestPath = Join-Path $InstallDir "speedtest.py"
    $uiPath = Join-Path $InstallDir "ui.py"
    $gitDir = Join-Path $InstallDir ".git"
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue | Select-Object -First 1
    $syncedByGit = $false

    if ((Test-Path $gitDir) -and $gitCmd) {
        Write-Info "检测到本地 Git 仓库（$InstallDir），正在更新..."
        try {
            & git -C $InstallDir pull --ff-only 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0 -and (Test-Path $speedtestPath) -and (Test-Path $uiPath)) {
                Write-Ok "已通过 Git 更新到最新版本。"
                $syncedByGit = $true
            }
        } catch {}
        if (-not $syncedByGit -and (Test-Path $speedtestPath) -and (Test-Path $uiPath)) {
            Write-Warn "Git 自动更新未完成（可能网络异常或存在本地修改），将继续使用本地文件。"
            $syncedByGit = $true
        }
    }

    if (-not $syncedByGit) {
        Write-Info "正在下载最新测速脚本到 $InstallDir ..."
        # 两个文件都下载成功后再一起替换，避免只更新了一个导致新旧版本混用
        $ok1 = Download-FileSafe -Url "$BaseUrl/speedtest.py" -Dest "$speedtestPath.new"
        $ok2 = $ok1 -and (Download-FileSafe -Url "$BaseUrl/ui.py" -Dest "$uiPath.new")
        if ($ok1 -and $ok2) {
            Move-Item -Path "$speedtestPath.new" -Destination $speedtestPath -Force
            Move-Item -Path "$uiPath.new" -Destination $uiPath -Force
        } else {
            Remove-Item -Path "$speedtestPath.new", "$uiPath.new" -Force -ErrorAction SilentlyContinue
        }
        if (-not ($ok1 -and $ok2)) {
            if ((Test-Path $speedtestPath) -and (Test-Path $uiPath)) {
                Write-Warn "在线更新失败（可能是 GitHub 网络连接超时），将继续使用本地已有版本。"
            } else {
                Write-Err "下载失败：$BaseUrl"
                Write-Host ""
                Write-Host "可能原因：当前网络无法直连 GitHub raw。" -ForegroundColor Yellow
                Write-Host "建议排查："
                Write-Host "  1. 检查网络/代理设置后重试"
                Write-Host "  2. 或打开浏览器下载仓库源码：https://github.com/oktothemoon/llm-speedtest"
                Write-Host "  3. 若有镜像源，可设置 `$env:LLM_SPEEDTEST_BASE_URL 后重试"
                return
            }
        } else {
            Write-Ok "脚本已就绪。"
        }
    }

    # --- 第 3 步：启动测速工具 ---
    Push-Location $InstallDir
    try {
        $pyArgs = @()
        if ($py.Prefix -and $py.Prefix.Count -gt 0) {
            $pyArgs += $py.Prefix
        }
        $pyArgs += $speedtestPath
        if ($ForwardArgs -and $ForwardArgs.Count -gt 0) {
            $pyArgs += $ForwardArgs
        }
        & $py.Exe @pyArgs
    } finally {
        Pop-Location
    }
}

Invoke-LlmSpeedtest @args
