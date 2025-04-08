from typing import Literal, Optional

import torch

PAD_VALUE = -10000

class LossFunc:
    """
        Loss function object that computes and aggregates the loss.
    """
    def __init__(self, reduction: Optional[Literal["mean", "sum"]] = "mean"):
        """Initializes the LossFunc object.

        Args:
            reduction: Aggregation function. Defaults to "mean".
        """
        self.reduction = reduction

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Computes the loss function values given the logits and labels associated to a set of documents.

        Args:
            logits (torch.Tensor): logits from the model associated to a set of documents
            labels (torch.Tensor): labels associated to the documents

        Returns:
            torch.Tensor: value of the loss function
        """
        raise NotImplementedError()

    def aggregate(self, loss: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Aggregates the loss values.

        Args:
            loss (torch.Tensor): loss values
            mask (Optional[torch.Tensor], optional): mask to apply to the loss values. Defaults to None.

        Returns:
            torch.Tensor: aggregated loss value
        """
        if self.reduction is None:
            return loss
        if mask is not None:
            loss = loss[~mask]
        if not loss.numel():
            return torch.tensor(0.0, requires_grad=True, device=loss.device)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unknown reduction {self.reduction}")

class RankNet(LossFunc):
    """
        RankNet loss proposed in \'Learning to Rank using Gradient Descent\' by Chris Burges et al.
    """
    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Computes the RankNet loss given the logits and labels.

        Args:
            logits (torch.Tensor): logits from the model associated to a set of documents
            labels (torch.Tensor): labels associated to the documents

        Returns:
            torch.Tensor: value of the RankNet loss
        """
        greater = labels[..., None] > labels[:, None]
        logits_mask = logits == PAD_VALUE
        label_mask = labels == PAD_VALUE
        mask = (
            logits_mask[..., None]
            | logits_mask[:, None]
            | label_mask[..., None]
            | label_mask[:, None]
            | ~greater
        )
        diff = logits[..., None] - logits[:, None]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            diff, greater.to(diff), reduction="none"
        )
        loss = loss.masked_fill(mask, 0)
        return self.aggregate(loss, mask)

class WeightedRankNet(LossFunc):
    """
        Variant of the RankNet loss proposed in \'FIRST: Faster Improved Listwise Reranking with Single Token Decoding\' by Revanth et al.
        This loss function prioritizes getting the ranks of higher-ranked candidates right over those of lower-ranked ones.
        It improves distillation.
    """
    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
            Computes the Weighted RankNet loss given the logits and labels.

        Args:
            logits (torch.Tensor): logits from the model associated to a set of documents
            labels (torch.Tensor): labels associated to the documents

        Returns:
            torch.Tensor: value of the Weighted RankNet loss
        """
        greater = labels[..., None] > labels[:, None]
        logits_mask = logits == PAD_VALUE
        label_mask = labels == PAD_VALUE
        mask = (
            logits_mask[..., None]
            | logits_mask[:, None]
            | label_mask[..., None]
            | label_mask[:, None]
            | ~greater
        )
        
        diff = logits[..., None] - logits[:, None]
       
        ranks = torch.argsort(logits, descending=True).argsort() + 1
        
        pairwise_rank_sum = ranks.unsqueeze(1) + ranks.unsqueeze(2)
        
        weight_matrix = 1 / pairwise_rank_sum  # matrix of inverse mean ranks
      
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            diff, greater.to(diff),  weight=weight_matrix, reduction="none"
        )
        
        loss = loss.masked_fill(mask, 0)
        
        return self.aggregate(loss, mask)

class LocalizedContrastive(LossFunc):
    """
        Localized Contrastive Loss (LCE) proposed.
    """
    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Computes the Localized Contrastive loss given the logits and labels.

        Args:
            logits (torch.Tensor): logits from the model associated to a set of documents
            labels (torch.Tensor): labels associated to the documents

        Returns:
            torch.Tensor: value of the LCE loss
        """
        mask = (labels == PAD_VALUE) | (logits == PAD_VALUE)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        labels = labels.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        return self.aggregate(loss)
