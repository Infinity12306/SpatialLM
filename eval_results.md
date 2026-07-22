# Evaluation Comparison

Notes:

- All SpatialLM-dataset rows use NMS@0.1 results.
- Layout metrics use the `micro` summary rows.
- Object metrics use the `micro` summary rows.
- Region metrics are reported at IoU@0.50 and IoU@0.75 in the eval logs. Baseline and GT-region stage2-only rows have no predicted region metrics, so region metrics are `-`.
- Best value in each metric column is **bold**; second best is <u>underlined</u>.

## ScanNet18 Test-Half

All rows use the same 156 scenes, 18 object classes, 2,174 GT objects, and common micro P/R/F1 evaluator. Values are copied from the specified logs; V-DETR uses NMS@0.25, while the two SpatialLM methods use NMS@0.1.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">NMS</th>
      <th rowspan="2">Pred</th>
      <th colspan="2">P-Object</th>
      <th colspan="2">R-Object</th>
      <th colspan="2">F1-Object</th>
    </tr>
    <tr>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SpatialLM Baseline</td>
      <td>0.10</td>
      <td>2,099</td>
      <td>0.6960</td>
      <td>0.5293</td>
      <td>0.6720</td>
      <td>0.5110</td>
      <td>0.6838</td>
      <td>0.5200</td>
    </tr>
    <tr>
      <td>Ours Hierarchical</td>
      <td>0.10</td>
      <td>2,129</td>
      <td><u>0.7252</u></td>
      <td><u>0.5857</u></td>
      <td><u>0.7102</u></td>
      <td><u>0.5736</u></td>
      <td><u>0.7176</u></td>
      <td><u>0.5796</u></td>
    </tr>
    <tr>
      <td>V-DETR</td>
      <td>0.25</td>
      <td>1,788</td>
      <td><strong>0.8848</strong></td>
      <td><strong>0.8048</strong></td>
      <td><strong>0.7277</strong></td>
      <td><strong>0.6619</strong></td>
      <td><strong>0.7986</strong></td>
      <td><strong>0.7264</strong></td>
    </tr>
  </tbody>
</table>

Sources:

- `V-DETR/logs/eval_half_test.log`
- `logs/scannet18/scannet18_ours_hierarchical_test_half/eval.log`
- `logs/scannet18/scannet18_spatiallm_baseline_test_half/eval.log`

## Recent Scorer-Conditioned Experiments

All values are object micro metrics after NMS@0.1. Joint-scorer rows pair the
SpatialLM model and `scorer.pt` from the same checkpoint.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">P-Object</th>
      <th colspan="2">R-Object</th>
      <th colspan="2">F1-Object</th>
    </tr>
    <tr>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline</td>
      <td>0.7528</td>
      <td>0.6229</td>
      <td><u>0.7106</u></td>
      <td>0.5880</td>
      <td>0.7311</td>
      <td>0.6049</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096</td>
      <td><u>0.7809</u></td>
      <td><u>0.6939</u></td>
      <td><strong>0.7198</strong></td>
      <td><strong>0.6396</strong></td>
      <td><strong>0.7491</strong></td>
      <td><u>0.6656</u></td>
    </tr>
    <tr>
      <td>Hier PredScorer</td>
      <td>0.7646</td>
      <td>0.6711</td>
      <td>0.7075</td>
      <td>0.6210</td>
      <td>0.7350</td>
      <td>0.6450</td>
    </tr>
    <tr>
      <td>Hier-20k ScorerFiltered</td>
      <td><strong>0.7815</strong></td>
      <td><strong>0.7061</strong></td>
      <td>0.7045</td>
      <td><u>0.6365</u></td>
      <td><u>0.7410</u></td>
      <td><strong>0.6695</strong></td>
    </tr>
    <tr>
      <td>Hier-20k ScorerFiltered Epoch2</td>
      <td>0.7133</td>
      <td>0.5983</td>
      <td>0.6662</td>
      <td>0.5588</td>
      <td>0.6890</td>
      <td>0.5779</td>
    </tr>
    <tr>
      <td>AttenScorer E2E TopK1536</td>
      <td>0.7003</td>
      <td>0.5949</td>
      <td>0.6398</td>
      <td>0.5436</td>
      <td>0.6687</td>
      <td>0.5681</td>
    </tr>
    <tr>
      <td>JointScorer Step12000</td>
      <td>0.7246</td>
      <td>0.6048</td>
      <td>0.6243</td>
      <td>0.5211</td>
      <td>0.6707</td>
      <td>0.5598</td>
    </tr>
    <tr>
      <td>JointScorer Final</td>
      <td>0.7089</td>
      <td>0.5918</td>
      <td>0.6346</td>
      <td>0.5297</td>
      <td>0.6697</td>
      <td>0.5590</td>
    </tr>
  </tbody>
</table>

## Region

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">P-Region</th>
      <th colspan="2">R-Region</th>
      <th colspan="2">F1-Region</th>
    </tr>
    <tr>
      <th>IoU@0.50</th>
      <th>IoU@0.75</th>
      <th>IoU@0.50</th>
      <th>IoU@0.75</th>
      <th>IoU@0.50</th>
      <th>IoU@0.75</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hier-20k</td>
      <td><strong>0.5764</strong></td>
      <td><strong>0.3747</strong></td>
      <td><strong>0.5681</strong></td>
      <td><strong>0.3693</strong></td>
      <td><strong>0.5722</strong></td>
      <td><strong>0.3720</strong></td>
    </tr>
    <tr>
      <td>Hier-20k GTRegion</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hier-40k</td>
      <td><u>0.5761</u></td>
      <td><u>0.3725</u></td>
      <td><u>0.5626</u></td>
      <td><u>0.3638</u></td>
      <td><u>0.5692</u></td>
      <td><u>0.3681</u></td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096</td>
      <td>0.5608</td>
      <td>0.3570</td>
      <td>0.5488</td>
      <td>0.3494</td>
      <td>0.5547</td>
      <td>0.3531</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max7200</td>
      <td>0.5660</td>
      <td>0.3652</td>
      <td>0.5488</td>
      <td>0.3542</td>
      <td>0.5573</td>
      <td>0.3596</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion Evict</td>
      <td>0.5692</td>
      <td>0.3588</td>
      <td>0.5543</td>
      <td>0.3494</td>
      <td>0.5617</td>
      <td>0.3540</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion BBoxMask</td>
      <td>0.5673</td>
      <td>0.3545</td>
      <td>0.5536</td>
      <td>0.3459</td>
      <td>0.5604</td>
      <td>0.3502</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion Evict</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion BBoxMask</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion Scorer</td>
      <td>0.5638</td>
      <td>0.3569</td>
      <td>0.5530</td>
      <td>0.3501</td>
      <td>0.5583</td>
      <td>0.3535</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion Scorer Thresh0.25</td>
      <td>0.5622</td>
      <td>0.3543</td>
      <td>0.5468</td>
      <td>0.3446</td>
      <td>0.5544</td>
      <td>0.3494</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion ScorerFiltered</td>
      <td>0.5688</td>
      <td>0.3557</td>
      <td>0.5488</td>
      <td>0.3432</td>
      <td>0.5586</td>
      <td>0.3493</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion ScorerFiltered NoisyFT Epoch2</td>
      <td>0.5636</td>
      <td>0.3535</td>
      <td>0.5516</td>
      <td>0.3459</td>
      <td>0.5575</td>
      <td>0.3497</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion AttentionOracle Keep75</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 PredRegion AttentionScorer E2E TopK1536</td>
      <td>0.5620</td>
      <td>0.3609</td>
      <td>0.5516</td>
      <td>0.3542</td>
      <td>0.5568</td>
      <td>0.3575</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion JointScorer Step12000</td>
      <td>0.5625</td>
      <td>0.3580</td>
      <td>0.5447</td>
      <td>0.3466</td>
      <td>0.5535</td>
      <td>0.3522</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion JointScorer Final</td>
      <td>0.5622</td>
      <td>0.3571</td>
      <td>0.5468</td>
      <td>0.3473</td>
      <td>0.5544</td>
      <td>0.3522</td>
    </tr>
  </tbody>
</table>

## Layout

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">P-Layout</th>
      <th colspan="2">R-Layout</th>
      <th colspan="2">F1-Layout</th>
    </tr>
    <tr>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline</td>
      <td><strong>0.9008</strong></td>
      <td><u>0.8782</u></td>
      <td><strong>0.8556</strong></td>
      <td><strong>0.8340</strong></td>
      <td><strong>0.8776</strong></td>
      <td><strong>0.8555</strong></td>
    </tr>
    <tr>
      <td>Hier-20k</td>
      <td>0.8964</td>
      <td>0.8756</td>
      <td><u>0.8426</u></td>
      <td><u>0.8231</u></td>
      <td><u>0.8686</u></td>
      <td><u>0.8486</u></td>
    </tr>
    <tr>
      <td>Hier-20k GTRegion</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hier-40k</td>
      <td>0.8978</td>
      <td>0.8772</td>
      <td>0.8332</td>
      <td>0.8140</td>
      <td>0.8643</td>
      <td>0.8444</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096</td>
      <td>0.8869</td>
      <td>0.8664</td>
      <td>0.8281</td>
      <td>0.8089</td>
      <td>0.8565</td>
      <td>0.8367</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max7200</td>
      <td>0.8888</td>
      <td>0.8691</td>
      <td>0.8193</td>
      <td>0.8012</td>
      <td>0.8526</td>
      <td>0.8338</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion Evict</td>
      <td>0.8976</td>
      <td>0.8748</td>
      <td>0.8257</td>
      <td>0.8046</td>
      <td>0.8601</td>
      <td>0.8382</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion BBoxMask</td>
      <td>0.8722</td>
      <td>0.8533</td>
      <td>0.8270</td>
      <td>0.8091</td>
      <td>0.8490</td>
      <td>0.8306</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion Evict</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion BBoxMask</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion Scorer</td>
      <td>0.8970</td>
      <td>0.8745</td>
      <td>0.8306</td>
      <td>0.8098</td>
      <td>0.8625</td>
      <td>0.8409</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion Scorer Thresh0.25</td>
      <td>0.8962</td>
      <td>0.8755</td>
      <td>0.8281</td>
      <td>0.8089</td>
      <td>0.8608</td>
      <td>0.8409</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion ScorerFiltered</td>
      <td>0.8906</td>
      <td>0.8693</td>
      <td>0.8115</td>
      <td>0.7920</td>
      <td>0.8492</td>
      <td>0.8288</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion ScorerFiltered NoisyFT Epoch2</td>
      <td>0.8989</td>
      <td><strong>0.8783</strong></td>
      <td>0.8267</td>
      <td>0.8077</td>
      <td>0.8613</td>
      <td>0.8415</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion AttentionOracle Keep75</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 PredRegion AttentionScorer E2E TopK1536</td>
      <td><u>0.9002</u></td>
      <td>0.8772</td>
      <td>0.8279</td>
      <td>0.8067</td>
      <td>0.8625</td>
      <td>0.8405</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion JointScorer Step12000</td>
      <td>0.8761</td>
      <td>0.8559</td>
      <td>0.8228</td>
      <td>0.8038</td>
      <td>0.8486</td>
      <td>0.8290</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion JointScorer Final</td>
      <td>0.8905</td>
      <td>0.8694</td>
      <td>0.8224</td>
      <td>0.8029</td>
      <td>0.8551</td>
      <td>0.8348</td>
    </tr>
  </tbody>
</table>

## Object

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">P-Object</th>
      <th colspan="2">R-Object</th>
      <th colspan="2">F1-Object</th>
    </tr>
    <tr>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
      <th>IoU@0.25</th>
      <th>IoU@0.50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline</td>
      <td>0.7528</td>
      <td>0.6229</td>
      <td>0.7106</td>
      <td>0.5880</td>
      <td>0.7311</td>
      <td>0.6049</td>
    </tr>
    <tr>
      <td>Hier-20k</td>
      <td>0.7736</td>
      <td>0.6734</td>
      <td>0.7217</td>
      <td>0.6282</td>
      <td>0.7468</td>
      <td>0.6500</td>
    </tr>
    <tr>
      <td>Hier-20k GTRegion</td>
      <td>0.7791</td>
      <td>0.6814</td>
      <td>0.7605</td>
      <td>0.6651</td>
      <td>0.7697</td>
      <td>0.6731</td>
    </tr>
    <tr>
      <td>Hier-40k</td>
      <td>0.7711</td>
      <td>0.6716</td>
      <td>0.7225</td>
      <td>0.6293</td>
      <td>0.7460</td>
      <td>0.6498</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096</td>
      <td>0.7809</td>
      <td>0.6939</td>
      <td>0.7198</td>
      <td>0.6396</td>
      <td>0.7491</td>
      <td>0.6656</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion</td>
      <td>0.7792</td>
      <td>0.7012</td>
      <td>0.7728</td>
      <td>0.6953</td>
      <td>0.7760</td>
      <td>0.6982</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max7200</td>
      <td>0.7833</td>
      <td>0.7026</td>
      <td>0.7023</td>
      <td>0.6299</td>
      <td>0.7406</td>
      <td>0.6642</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion Evict</td>
      <td>0.7754</td>
      <td>0.6801</td>
      <td>0.7175</td>
      <td>0.6293</td>
      <td>0.7454</td>
      <td>0.6537</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion BBoxMask</td>
      <td><u>0.8396</u></td>
      <td><u>0.7534</u></td>
      <td>0.7642</td>
      <td>0.6856</td>
      <td><u>0.8001</u></td>
      <td><u>0.7179</u></td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion Evict</td>
      <td>0.7698</td>
      <td>0.6882</td>
      <td><u>0.7850</u></td>
      <td><u>0.7017</u></td>
      <td>0.7773</td>
      <td>0.6949</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion BBoxMask</td>
      <td><strong>0.8443</strong></td>
      <td><strong>0.7664</strong></td>
      <td><strong>0.8335</strong></td>
      <td><strong>0.7567</strong></td>
      <td><strong>0.8389</strong></td>
      <td><strong>0.7615</strong></td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion Scorer</td>
      <td>0.7646</td>
      <td>0.6711</td>
      <td>0.7075</td>
      <td>0.6210</td>
      <td>0.7350</td>
      <td>0.6450</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion Scorer Thresh0.25</td>
      <td>0.7426</td>
      <td>0.6457</td>
      <td>0.6740</td>
      <td>0.5860</td>
      <td>0.7066</td>
      <td>0.6144</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion ScorerFiltered</td>
      <td>0.7815</td>
      <td>0.7061</td>
      <td>0.7045</td>
      <td>0.6365</td>
      <td>0.7410</td>
      <td>0.6695</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion ScorerFiltered NoisyFT Epoch2</td>
      <td>0.7133</td>
      <td>0.5983</td>
      <td>0.6662</td>
      <td>0.5588</td>
      <td>0.6890</td>
      <td>0.5779</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 GTRegion AttentionOracle Keep75</td>
      <td>0.7768</td>
      <td>0.6955</td>
      <td>0.7694</td>
      <td>0.6890</td>
      <td>0.7731</td>
      <td>0.6922</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 PredRegion AttentionScorer E2E TopK1536</td>
      <td>0.7003</td>
      <td>0.5949</td>
      <td>0.6398</td>
      <td>0.5436</td>
      <td>0.6687</td>
      <td>0.5681</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion JointScorer Step12000</td>
      <td>0.7246</td>
      <td>0.6048</td>
      <td>0.6243</td>
      <td>0.5211</td>
      <td>0.6707</td>
      <td>0.5598</td>
    </tr>
    <tr>
      <td>Hier-20k Res16 Max4096 PredRegion JointScorer Final</td>
      <td>0.7089</td>
      <td>0.5918</td>
      <td>0.6346</td>
      <td>0.5297</td>
      <td>0.6697</td>
      <td>0.5590</td>
    </tr>
  </tbody>
</table>

Sources:

- `logs/eval_results/baseline_test_NMS_0.1.log`
- `logs/eval_results/hier_20000_test_NMS_0.1.log`
- `logs/eval_results/hier_20000_stage2_ckpt_12000_gt_regions_NMS_0.1.log`
- `logs/eval_results/hier_40000_test_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_stage1_9996_stage2_14000_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max7200_stage1_9996_stage2_14000_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_evict_pred_regions_stage1_9996_stage2_14392_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_bbox_mask_pred_regions_stage1_9996_stage2_14392_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_gt_regions_stage2_14392_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_gt_regions_evict_stage2_14392_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_gt_regions_bbox_mask_stage2_14392_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_scorer_pred_regions_stage1_9996_stage2_14392_scorer_29488_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_scorer_pred_regions_stage1_9996_stage2_14392_scorer_29488_threshold_0.25_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_scorer_filtered_point_token_pred_regions_stage1_9996_stage2_14392_scorer_29488_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_scorer_filtered_point_token_restart_epoch2_pred_regions_stage1_9996_stage2_28784_scorer_29488_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_attention_oracle_gt_regions_stage2_14392_keep75_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_attention_scorer_e2e_hardtopk1536_pred_regions_stage1_9996_stage2_14392_scorer_40000_max3200_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_joint_scorer_12000_pred_regions_stage1_9996_NMS_0.1.log`
- `logs/eval_results/hier_20000_res16_max4096_joint_scorer_final_pred_regions_stage1_9996_NMS_0.1.log`
