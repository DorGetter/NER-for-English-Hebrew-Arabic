import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as tud
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
from seqeval.metrics import accuracy_score
from tqdm import trange
import pandas as pd


def valid_loss_decreases(valid_los_fresh, valid_los_old, valid_los_oldest):
    #  if 2 epochs before the last epoch not decreasing exit sends False to make the machine exit
    if (valid_los_fresh is not None) and (valid_los_old is None or valid_los_oldest is None):
        return True
    else:
        if (valid_los_old is not None) and (valid_los_fresh < valid_los_old):
            return True
        elif (valid_los_oldest is not None) and (valid_los_fresh < valid_los_oldest):
            return True
        elif (valid_los_oldest is not None and valid_los_old is not None) and (valid_los_old < valid_los_oldest):
            return True
        elif (valid_los_old is None) or (valid_los_oldest is None):
            return True
        else:
            return False


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    train_dataloader: tud.DataLoader,
    valid_dataloader: tud.DataLoader,
    tag_values,
    opts: argparse.Namespace,
    checkpoints: Path,
):

    train_loss, validation_loss, validation_accuracy = [], [], []
    validation_max_accuracy = 0.0
    i = 0
    validation_loss_old = None
    validation_loss_oldest = None
    data = pd.DataFrame(
        columns=[
            "Epoch no",
            "Accuracy",
            "Precision-Weighted",
            "Recall-Weighted",
            "F1-Score-Weighted",
            "Precision-Macro",
            "Recall-Macro",
            "F1-Score-Macro",
            "Precision-Micro",
            "Recall-Micro",
            "F1-Score-Micro",
            "Valid Loss",
            "Train Loss",
        ]
    )
    for _ in trange(opts.num_epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        i += 1
        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(opts.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opts.max_grad_norm)
            # update parameters
            optimizer.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {:.3f}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        train_loss.append(avg_train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(opts.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss.append(eval_loss)
        print("Validation loss: {:.3f}".format(eval_loss))
        pred_tags = [
            tag_values[p_i]
            for p, l in zip(predictions, true_labels)
            for p_i, l_i in zip(p, l)
            if tag_values[l_i] != "PAD"
        ]
        valid_tags = [tag_values[l_i] for l in true_labels for l_i in l if tag_values[l_i] != "PAD"]
        epoch_accuracy = 100.0 * accuracy_score(pred_tags, valid_tags)
        validation_accuracy.append(epoch_accuracy)
        # ========================================
        #               Statistics
        # ========================================
        validation_loss_oldest = validation_loss_old
        validation_loss_old = eval_loss
        precision_macro = precision_recall_fscore_support(pred_tags, valid_tags, average="macro")
        precision_micro = precision_recall_fscore_support(pred_tags, valid_tags, average="micro")
        precision_weighted = precision_recall_fscore_support(pred_tags, valid_tags, average="weighted")
        data.loc[i] = [
            i,
            "{:.3f}".format(100.0 * accuracy_score(pred_tags, valid_tags)),
            "{:.3f}".format(100.0 * precision_weighted[0]),
            "{:.3f}".format(100.0 * precision_weighted[1]),
            "{:.3f}".format(100.0 * precision_weighted[2]),
            "{:.3f}".format(100.0 * precision_macro[0]),
            "{:.3f}".format(100.0 * precision_macro[1]),
            "{:.3f}".format(100.0 * precision_macro[2]),
            "{:.3f}".format(100.0 * precision_micro[0]),
            "{:.3f}".format(100.0 * precision_micro[1]),
            "{:.3f}".format(100.0 * precision_micro[2]),
            "{:.3f}".format(100.0 * eval_loss),
            "{:.3f}".format(100.0 * avg_train_loss),
        ]
        data.to_csv("parameters.csv")
        data.to_pickle("pickle_parameters.pkl")
        print("Validation Accuracy: {:.3f}".format(epoch_accuracy))
        if not valid_loss_decreases(eval_loss, validation_loss_old, validation_loss_oldest):
            # Update the learning rate.
            scheduler.step()
        if validation_max_accuracy < epoch_accuracy:
            validation_max_accuracy = epoch_accuracy
            torch.save(model, checkpoints.joinpath("model.pth"))

    return {"train loss": train_loss, "validation loss": validation_loss, "validation accuracy": validation_accuracy}
