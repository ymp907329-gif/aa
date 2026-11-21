# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
    best_psnr = 0.0
    patience = 8  # 连续 8 个 epoch 无提升就停
    no_improve = 0

    for epoch in range(1, cfg['train']['epochs'] + 1):
        train_loss = train_one_epoch(...)
        scheduler.step()

        psnr, ssim = validate(...)
        print(f'[Epoch {epoch}] loss={train_loss:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}')

        if psnr > best_psnr:
            best_psnr = psnr
            no_improve = 0
            save_checkpoint({...}, cfg['train']['save_dir'], 'best.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stop at epoch {epoch}. Best PSNR: {best_psnr:.2f}')
                break


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
