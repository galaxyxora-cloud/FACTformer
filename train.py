import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from data import train_dataloader
from utils import Adder, tensor_metric
from valid import _valid


def _radial_weight(h, w, alpha=0.75, device='cpu'):
    fy = torch.fft.fftfreq(h, d=1.0).to(device)  # [-0.5, 0.5)
    fx = torch.fft.rfftfreq(w, d=1.0).to(device)  # [0, 0.5]
    gy, gx = torch.meshgrid(fy, fx, indexing='ij')
    r = torch.sqrt(gx ** 2 + gy ** 2)
    r = r / r.max().clamp_min(1e-6)
    return (r ** alpha)


class FreqL1Loss(torch.nn.Module):

    def __init__(self, eps=1e-3, delta=1e-6, alpha=0.75):
        super().__init__()
        self.eps = eps
        self.delta = delta
        self.alpha = alpha

    def forward(self, x, y):
        N, C, H, W = x.shape
        X = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        Y = torch.fft.rfft2(y, dim=(-2, -1), norm='ortho')

        ax = torch.abs(X)
        ay = torch.abs(Y)

        diff = torch.log(ax + self.delta) - torch.log(ay + self.delta)
        w = _radial_weight(H, W, alpha=self.alpha, device=x.device)
        diff = diff * w

        return torch.abs(diff).mean()


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    charb_eps = float(getattr(args, 'charb_eps', 1e-3))

    criterion_pix = torch.nn.L1Loss()
    criterion_freq = FreqL1Loss(eps=charb_eps, delta=1e-6, alpha=0.75)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)

    warmup_epochs = max(5, min(200, int(args.num_epoch * 0.03)))
    scheduler_warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.num_epoch - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

    start_epoch = 1
    best_score = -1
    best_psnr = -1
    best_ssim = -1

    checkpoint_path = _get_checkpoint_path(args)
    if checkpoint_path:
        start_epoch, best_score, best_psnr, best_ssim = _load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )

    global_step_start = (start_epoch - 1) * max_iter
    log_dir = getattr(args, 'log_dir', None)

    def _has_tb_events(d):
        try:
            return any(fn.startswith('events.out.tfevents') for fn in os.listdir(d))
        except Exception:
            return False

    if log_dir:
        if checkpoint_path:
            writer = SummaryWriter(log_dir=log_dir, purge_step=global_step_start)
        else:
            if os.path.exists(log_dir) and _has_tb_events(log_dir):
                from datetime import datetime
                run_subdir = datetime.now().strftime('run_%Y%m%d-%H%M%S')
                log_dir = os.path.join(log_dir, run_subdir)
            writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = SummaryWriter(purge_step=global_step_start) if checkpoint_path else SummaryWriter()

    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()

    epoch_psnr_adder = Adder()
    epoch_ssim_metric_adder = Adder()
    iter_psnr_adder = Adder()
    iter_ssim_metric_adder = Adder()

    for epoch_idx in range(start_epoch, args.num_epoch + 1):
        model.train()

        for iter_idx, (input_img, label_img) in enumerate(dataloader):
            input_img = input_img.to(device, non_blocking=True)
            label_img = label_img.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_full = model(input_img)  # (N,C,H,W)

            pred_full = torch.nan_to_num(pred_full, nan=0.0, posinf=1.0, neginf=0.0)

            label_1x = torch.clamp(torch.nan_to_num(label_img, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            label_1_2 = F.interpolate(label_1x, scale_factor=0.5, mode='bilinear', align_corners=False)
            label_1_4 = F.interpolate(label_1x, scale_factor=0.25, mode='bilinear', align_corners=False)

            pred_1x = pred_full
            pred_1_2 = F.interpolate(pred_full, scale_factor=0.5, mode='bilinear', align_corners=False)
            pred_1_4 = F.interpolate(pred_full, scale_factor=0.25, mode='bilinear', align_corners=False)

            pix_s = criterion_pix(pred_1_4, label_1_4)  # 1/4
            pix_m = criterion_pix(pred_1_2, label_1_2)  # 1/2
            pix_l = criterion_pix(pred_1x, label_1x)  # 1x
            loss_pix = pix_s + pix_m + pix_l

            loss_fft = criterion_freq(pred_1x, label_1x)

            loss = loss_pix + 0.1 * loss_fft

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            iter_pixel_adder(loss_pix.item());
            epoch_pixel_adder(loss_pix.item())
            iter_fft_adder(loss_fft.item());
            epoch_fft_adder(loss_fft.item())

            psnr, ssim = _compute_metrics(pred_1x, label_1x)
            iter_psnr_adder(psnr);
            epoch_psnr_adder(psnr)
            iter_ssim_metric_adder(ssim);
            epoch_ssim_metric_adder(ssim)

            if (iter_idx + 1) % args.print_freq == 0:
                total_loss = iter_pixel_adder.average() + 0.1 * iter_fft_adder.average()
                avg_psnr = iter_psnr_adder.average()
                avg_ssim = iter_ssim_metric_adder.average()
                score = avg_psnr + 100.0 * avg_ssim
                current_lr = optimizer.param_groups[0]['lr']

                print(
                    f"E{epoch_idx:03d}[{iter_idx + 1:4d}/{max_iter}] "
                    f"LR:{current_lr:.2e} "
                    f"Loss:{total_loss:.4f} "
                    f"PSNR:{avg_psnr:.2f} SSIM:{avg_ssim:.4f} "
                    f"Score(PSNR+100*SSIM):{score:.2f} "
                    f"(Pixel L1, Freq L1)"
                )

                global_step = iter_idx + (epoch_idx - 1) * max_iter
                writer.add_scalar('Train/Loss_total', total_loss, global_step)
                writer.add_scalar('Train/Loss_pixel_l1', iter_pixel_adder.average(), global_step)
                writer.add_scalar('Train/Loss_fft_l1_x0.1', 0.1 * iter_fft_adder.average(), global_step)
                writer.add_scalar('Train/PSNR', avg_psnr, global_step)
                writer.add_scalar('Train/SSIM', avg_ssim, global_step)
                writer.add_scalar('Train/Score_PSNR_plus_10xSSIM', score, global_step)
                writer.add_scalar('Train/LR', current_lr, global_step)

                iter_pixel_adder.reset()
                iter_fft_adder.reset()
                iter_psnr_adder.reset()
                iter_ssim_metric_adder.reset()

        _save_latest_checkpoint(model, optimizer, scheduler, epoch_idx,
                                best_score, best_psnr, best_ssim, args)
        if epoch_idx % (args.save_freq) == 0:
            _save_periodic_checkpoint(model, optimizer, scheduler, epoch_idx,
                                      best_score, best_psnr, best_ssim, args)
        _save_model_weights(model, args.model_save_dir, 'model.pkl')

        if epoch_idx % args.valid_freq == 0:
            val_psnr, val_ssim = _valid(model, args, epoch_idx)
            val_score = val_psnr + 10.0 * val_ssim

            print(f"[Val] PSNR:{val_psnr:.2f} SSIM:{val_ssim:.4f} Score(PSNR+10*SSIM):{val_score:.2f}")

            writer.add_scalar('Val/PSNR', val_psnr, epoch_idx)
            writer.add_scalar('Val/SSIM', val_ssim, epoch_idx)
            writer.add_scalar('Val/Score_PSNR_plus_10xSSIM', val_score, epoch_idx)

            # 记录最佳
            if val_score > best_score:
                best_score = val_score
                best_psnr = val_psnr
                best_ssim = val_ssim
                _save_model_weights(model, args.model_save_dir, 'Best.pkl')
                print(f"[Best] Score:{best_score:.2f} (PSNR:{best_psnr:.2f} SSIM:{best_ssim:.4f})")

        epoch_pixel_adder.reset()
        epoch_fft_adder.reset()
        epoch_psnr_adder.reset()
        epoch_ssim_metric_adder.reset()
        scheduler.step()

    _save_final_model(model, optimizer, scheduler, best_score, best_psnr, best_ssim, args)
    print(f"[Done] Best Score:{best_score:.2f} PSNR:{best_psnr:.2f} SSIM:{best_ssim:.4f}")
    writer.close()


def _compute_metrics(pred_img, target_img):
    with torch.no_grad():
        pred_img = torch.nan_to_num(pred_img, nan=0.0, posinf=1.0, neginf=0.0)
        target_img = torch.nan_to_num(target_img, nan=0.0, posinf=1.0, neginf=0.0)
        pred_img_clamped = torch.clamp(pred_img, 0.0, 1.0)
        target_img_clamped = torch.clamp(target_img, 0.0, 1.0)

        psnr = tensor_metric(pred_img_clamped, target_img_clamped, 'PSNR', data_range=1)
        ssim = tensor_metric(pred_img_clamped, target_img_clamped, 'SSIM', data_range=1)
    return psnr, ssim


def _get_checkpoint_path(args):
    if getattr(args, 'resume', None) and os.path.exists(args.resume):
        print(f"[Resume] {args.resume}")
        return args.resume

    if os.path.exists(args.model_save_dir):
        checkpoint_files = glob.glob(os.path.join(args.model_save_dir, 'checkpoint_*.pth'))
        latest_ckpt = os.path.join(args.model_save_dir, 'latest_checkpoint.pth')
        if os.path.exists(latest_ckpt):
            checkpoint_files.append(latest_ckpt)
        if checkpoint_files:
            latest = max(checkpoint_files, key=os.path.getmtime)
            print(f"[Resume] {latest}")
            return latest

    print("[Train] Starting from scratch")
    return None


def _load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    import numpy as np
    from torch.serialization import add_safe_globals

    add_safe_globals([np.core.multiarray.scalar])

    def _strip_module_prefix(sd):
        if not isinstance(sd, dict):
            return sd
        need_strip = any(k.startswith('module.') for k in sd.keys())
        return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}

    try:
        state = torch.load(checkpoint_path, map_location='cpu')  # PyTorch 2.6 默认 weights_only=True
    except Exception:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    start_epoch = 1
    best_score = -1
    best_psnr = -1
    best_ssim = -1

    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(_strip_module_prefix(state['model']), strict=False)
        if 'optimizer' in state and isinstance(state['optimizer'], dict):
            try:
                optimizer.load_state_dict(state['optimizer'])
            except Exception:
                pass
        if 'scheduler' in state and isinstance(state['scheduler'], dict):
            try:
                scheduler.load_state_dict(state['scheduler'])
            except Exception:
                pass

        start_epoch = int(state.get('epoch', 1))
        best_score = float(state.get('best_score', -1))
        best_psnr = float(state.get('best_psnr', -1))
        best_ssim = float(state.get('best_ssim', -1))

    elif isinstance(state, dict):
        model.load_state_dict(_strip_module_prefix(state), strict=False)
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(state)}")

    print(f"[Loaded] Epoch:{start_epoch} Score:{best_score:.2f} PSNR:{best_psnr:.2f} SSIM:{best_ssim:.4f}")
    return start_epoch, best_score, best_psnr, best_ssim


def _save_latest_checkpoint(model, optimizer, scheduler, epoch_idx,
                            best_score, best_psnr, best_ssim, args):
    os.makedirs(args.model_save_dir, exist_ok=True)
    torch.save({
        'epoch': epoch_idx + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_score': best_score,
        'best_psnr': best_psnr,
        'best_ssim': best_ssim
    }, os.path.join(args.model_save_dir, 'latest_checkpoint.pth'))


def _save_periodic_checkpoint(model, optimizer, scheduler, epoch_idx,
                              best_score, best_psnr, best_ssim, args):
    os.makedirs(args.model_save_dir, exist_ok=True)
    torch.save({
        'epoch': epoch_idx + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_score': best_score,
        'best_psnr': best_psnr,
        'best_ssim': best_ssim
    }, os.path.join(args.model_save_dir, f'checkpoint_{epoch_idx}.pth'))


def _save_model_weights(model, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'model': model.state_dict()}, os.path.join(save_dir, filename))


def _save_final_model(model, optimizer, scheduler, best_score, best_psnr, best_ssim, args):
    _save_model_weights(model, args.model_save_dir, 'Final.pkl')
    torch.save({
        'epoch': args.num_epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_score': best_score,
        'best_psnr': best_psnr,
        'best_ssim': best_ssim
    }, os.path.join(args.model_save_dir, 'final_checkpoint.pth'))