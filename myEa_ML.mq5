#include <Trade\Trade.mqh>

// CTradeオブジェクトを作成
CTrade trade;
int lastTrend = 0;
// グローバル変数: ONNXモデルハンドルとインジケータハンドル
long onnx_handle = INVALID_HANDLE;
int ma10_handle = INVALID_HANDLE;
int ma50_handle = INVALID_HANDLE;
int bb_upper_handle = INVALID_HANDLE;
int bb_lower_handle = INVALID_HANDLE;
int rsi_handle = INVALID_HANDLE;
int macd_handle = INVALID_HANDLE;
int atr_handle = INVALID_HANDLE;
int stoch_k_handle = INVALID_HANDLE;
int stoch_d_handle = INVALID_HANDLE;
int adx_handle = INVALID_HANDLE;


// ハンドル初期化関数
bool InitializeHandles() {
    if (ma10_handle == INVALID_HANDLE)
        ma10_handle = iMA(NULL, 0, 10, 0, MODE_SMA, PRICE_CLOSE);

    if (ma50_handle == INVALID_HANDLE)
        ma50_handle = iMA(NULL, 0, 50, 0, MODE_SMA, PRICE_CLOSE);

    if (bb_upper_handle == INVALID_HANDLE)
        bb_upper_handle = iBands(NULL, 0, 20, 2, 0, PRICE_CLOSE);

    if (bb_lower_handle == INVALID_HANDLE)
        bb_lower_handle = iBands(NULL, 0, 20, 2, 0, PRICE_CLOSE);

    if (rsi_handle == INVALID_HANDLE)
        rsi_handle = iRSI(NULL, 0, 14, PRICE_CLOSE);

    if (atr_handle == INVALID_HANDLE)
        atr_handle = iATR(NULL, 0, 14);

      if (adx_handle == INVALID_HANDLE) {
          adx_handle = iADX(NULL, 0, 14);  // シンボル: NULL, 時間足: 0, 期間: 14
      }


    if (ma10_handle == INVALID_HANDLE || ma50_handle == INVALID_HANDLE || bb_upper_handle == INVALID_HANDLE || 
        bb_lower_handle == INVALID_HANDLE || rsi_handle == INVALID_HANDLE || atr_handle == INVALID_HANDLE || adx_handle == INVALID_HANDLE) {
        Print("Failed to initialize indicator handles. Error: ", GetLastError());
        return false;
    }
    return true;
}


// ハンドル解放関数
void ReleaseHandles() {
    if (ma10_handle != INVALID_HANDLE) {
        IndicatorRelease(ma10_handle);
        ma10_handle = INVALID_HANDLE;
    }
    if (ma50_handle != INVALID_HANDLE) {
        IndicatorRelease(ma50_handle);
        ma50_handle = INVALID_HANDLE;
    }
    if (bb_upper_handle != INVALID_HANDLE) {
        IndicatorRelease(bb_upper_handle);
        bb_upper_handle = INVALID_HANDLE;
    }
    if (bb_lower_handle != INVALID_HANDLE) {
        IndicatorRelease(bb_lower_handle);
        bb_lower_handle = INVALID_HANDLE;
    }
    if (rsi_handle != INVALID_HANDLE) {
        IndicatorRelease(rsi_handle);
        rsi_handle = INVALID_HANDLE;
    }
    if (macd_handle != INVALID_HANDLE) {
        IndicatorRelease(macd_handle);
        macd_handle = INVALID_HANDLE;
    }
    if (atr_handle != INVALID_HANDLE) {
        IndicatorRelease(atr_handle);
        atr_handle = INVALID_HANDLE;
    }
    if (stoch_k_handle != INVALID_HANDLE) {
        IndicatorRelease(stoch_k_handle);
        stoch_k_handle = INVALID_HANDLE;
    }
    if (stoch_d_handle != INVALID_HANDLE) {
        IndicatorRelease(stoch_d_handle);
        stoch_d_handle = INVALID_HANDLE;
    }
    if (adx_handle != INVALID_HANDLE){
        IndicatorRelease(adx_handle);
        adx_handle = INVALID_HANDLE;
    }
}
bool PrepareFeatures(float &input_data[][15]) {
    // 動的配列を用意してインジケータの値を取得
    double ma10_values[], ma50_values[], bb_upper_values[], bb_lower_values[], rsi_values[], atr_values[], adx_values[];

    double close = iClose(NULL, 0, 0);
    if (CopyBuffer(ma10_handle, 0, 0, 1, ma10_values) <= 0 ||
        CopyBuffer(ma50_handle, 0, 0, 1, ma50_values) <= 0 ||
        CopyBuffer(bb_upper_handle, 1, 0, 1, bb_upper_values) <= 0 ||
        CopyBuffer(bb_lower_handle, 2, 0, 1, bb_lower_values) <= 0 ||
        CopyBuffer(rsi_handle, 0, 0, 1, rsi_values) <= 0 ||
        CopyBuffer(atr_handle, 0, 0, 1, atr_values) <= 0 ||
        CopyBuffer(adx_handle, 0, 0, 1, adx_values) <= 0) {
        Print("Error retrieving indicator data. Error: ", GetLastError());
        return false;
    }

    // データをinput_dataに設定
    input_data[0][0] = (float)close;
    input_data[0][1] = (float)ma10_values[0];
    input_data[0][2] = (float)ma50_values[0];
    input_data[0][3] = (float)rsi_values[0];
    //double prev_close = iClose(NULL, 0, 288);  // 1期間前の終値
    //input_data[0][4] = (float)((close - prev_close) / prev_close);
    input_data[0][4] = (float)bb_upper_values[0];
    input_data[0][5] = (float)bb_lower_values[0];
    input_data[0][6] = (float)atr_values[0];
    input_data[0][7] = (float)(bb_upper_values[0] - bb_lower_values[0]);
    input_data[0][8] = (float)adx_values[0];
    double high = iHigh(NULL,0,0);
    double low = iLow(NULL,0,0);
    input_data[0][9] = (float)(high-low);
    input_data[0][10] = (float)(close - iClose(NULL,0,10));
    input_data[0][11] = (float)(close - iClose(NULL,0,50));
    input_data[0][12] = (float)(close - iClose(NULL,0,100));
    input_data[0][13] = (float)(close - iClose(NULL,0,200));
    input_data[0][14] = (float)(close - iClose(NULL,0,300));
    return true;
}
double PerformPrediction() {
    float input_data[1][15];
    //if (!PrepareFeatures(input_data)) {
    //    Print("特徴量の準備に失敗しました。");
    //    return 0.5;
    //}

    // 入力形状の設定
    long input_shape[] = {1, 15}; // バッチサイズ1、特徴量8
    if (!OnnxSetInputShape(onnx_handle, 0, input_shape)) {
        Print("OnnxSetInputShape failed, error ", GetLastError());
        return 0.5;
    }

    // 出力1（label）の形状を設定
    long output_shape1[] = {1}; // バッチサイズ1、1つのラベル
    if (!OnnxSetOutputShape(onnx_handle, 0, output_shape1)) {
        Print("OnnxSetOutputShape for label failed, error ", GetLastError());
        return 0.5;
    }

    // 出力2（probabilities）の形状を設定
    long output_shape2[] = {1, 3}; // バッチサイズ1、クラス数2
    if (!OnnxSetOutputShape(onnx_handle, 1, output_shape2)) {
        Print("OnnxSetOutputShape for probabilities failed, error ", GetLastError());
        return 0.5;
    }

    // 入力と出力のデータ配列
    vectorf input_v(15);        // 入力データ（float型）
    vector output_label(1);    // 出力データ（label、double型）
    matrix probabilities(1, 3); // 出力データ（probabilities、double型）

    // 入力データを設定
    for (int i = 0; i < 15; i++) {
        input_v[i] = input_data[0][i];
    }

    // モデルの実行
    if (!OnnxRun(onnx_handle, ONNX_DEBUG_LOGS, input_v, output_label, probabilities)) {
        Print("OnnxRun failed, error ", GetLastError());
        return 0.5;
    }

    // 結果の表示
    Print("Predicted label: ", output_label[0]);
    Print("Probabilities: [", probabilities[0][0], ", ", probabilities[0][1], "]");
    if(probabilities[0][(int)output_label[0]] > 0.6){
       // ラベルを返却
       return (double)output_label[0];
    }else{
      return 1;
    }
}



// EAの初期化
int OnInit() {
   Print("start Init");
    if (!InitializeHandles()) {
        Print("Failed to initialize indicator handles.");
        return(INIT_FAILED);
    }
    Print("Before OnnxCreate");
    onnx_handle = OnnxCreate("xgboost_model.onnx", 0);
    Print("After OnnxCreate");
    if (onnx_handle == INVALID_HANDLE) {
        Print("Failed to load ONNX model. Error: ", GetLastError());
        return(INIT_FAILED);
    }
    Print("Init Succeeded");
    EventSetTimer(60*5); 
    return(INIT_SUCCEEDED);
}

// タイマーイベント（5分ごとに呼び出される）
void OnTimer() {
Print("OnTimer");
    double prediction = PerformPrediction();
    double real_prediction = prediction;  // ターゲット値のインデックスを指定
    // Ask価格を取得
    double ask_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    // Bid価格を取得
    double bid_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double sl = 0;
    double tp = 0;
    double price = 0;
    double lot = 0;
    if(prediction >=  1.8){
      //buy
      price = ask_price;
      //tp = (real_prediction + ask_price)/2.0;
      tp = ask_price + 600*_Point;
      sl = bid_price - 600*_Point;
      lot = CalculateLot(1);
      if(lastTrend == -1)CloseAllPositions();
      lastTrend = 1;
      PlaceOrder(ORDER_TYPE_BUY,lot,price,sl,tp);

    }else if(prediction <= 0.2){
      price = bid_price;
      //tp = (real_prediction+bid_price)/2.0;
      tp = bid_price - 600*_Point;
      sl = ask_price + 600*_Point;
      lot = CalculateLot(1);
      if(lastTrend == 1)CloseAllPositions();
      lastTrend = -1;
      PlaceOrder(ORDER_TYPE_SELL,lot,price,sl,tp);

      

    }else{
    }
    //PrintFormat("■Prediction: %.2f, Ask: %.5f, Bid: %.5f", prediction, ask_price, bid_price);
    /*
    Print("Prediction result (real price): ", real_prediction);
    double ask_price = 0.0;
    double bid_price = 0.0;
    double close = 0.0;
    close = iClose(NULL,0,0);
      Print("close: ",close);
    // 現在の価格を取得
    if (SymbolInfoDouble(Symbol(), SYMBOL_ASK, ask_price) && 
        SymbolInfoDouble(Symbol(), SYMBOL_BID, bid_price)) {
        Print("Current Ask price: ", ask_price);
        Print("Current Bid price: ", bid_price);
    } else {
        Print("Failed to retrieve Ask and Bid prices. Error: ", GetLastError());
    }
    ]
    */
    Print("End OnTimer");
}


// 終了処理
void OnDeinit(const int reason) {
    ReleaseHandles();
    if (onnx_handle != INVALID_HANDLE) {
        OnnxRelease(onnx_handle);
        onnx_handle = INVALID_HANDLE;
    }

    EventKillTimer();  // タイマー解除
}
//+------------------------------------------------------------------+
//| 新規注文を出す関数                                              |
//| type: 注文タイプ（BUY or SELL）                                  |
//| lot: ロット数                                                   |
//| price: 注文価格（成行注文の場合は現在価格を使用）                |
//| sl: ストップロス価格                                            |
//| tp: 利益確定（テイクプロフィット）価格                           |
//+------------------------------------------------------------------+
void PlaceOrder(ENUM_ORDER_TYPE type, double lot, double price, double sl, double tp)
{
   if(lot <= 0)
   {
      Print("ロット数が無効です。");
      return;
   }
   
   // 買い注文
   if(type == ORDER_TYPE_BUY)
   {
      if(trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, lot, price, sl, tp))
      {
         //Print("買い注文が成功しました。");
      }
      else
      {
         //PrintFormat("買い注文に失敗しました。エラー: %d", GetLastError());
      }
   }
   // 売り注文
   else if(type == ORDER_TYPE_SELL)
   {
      if(trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, lot, price, sl, tp))
      {
         //Print("売り注文が成功しました。");
      }
      else
      {
         //PrintFormat("売り注文に失敗しました。エラー: %d", GetLastError());
      }
   }
   else
   {
      Print("無効な注文タイプです。");
   }
}
//+------------------------------------------------------------------+
//| 証拠金の1%に基づいてロット数を計算する関数                     |
//| risk_percentage: リスク割合（例: 1% = 0.01）                     |
//| Returns: ロット数                                                |
//+------------------------------------------------------------------+
double CalculateLot(double risk_percentage)
{
    // 変数の初期化
    double free_margin = AccountInfoDouble(ACCOUNT_FREEMARGIN);  // 余剰証拠金
    double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);        // 現在のAsk価格
    long leverage = AccountInfoInteger(ACCOUNT_LEVERAGE);      // レバレッジ倍率
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE); // 1ティックの価値
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP); // ロットステップ

    // 取引シンボルが正常に設定されているか確認
    if(price <= 0 || leverage <= 0 || tick_value <= 0)
    {
        Print("価格、レバレッジ、ティック値が無効です。");
        return 0.0;
    }

    // 証拠金の1%を計算
    double risk_amount = free_margin * risk_percentage;

    // 1ロットあたりの必要証拠金を計算
    double margin_per_lot = (price * 100000) / (double)leverage;

    // ロット数を計算
    double lot = risk_amount / margin_per_lot;

    // ロット数を取引可能なステップに合わせて丸める
    lot = MathRound(lot / lot_step) * lot_step;

    // 最小および最大ロット制限のチェック
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

    if(lot < min_lot) lot = min_lot;  // 最小ロット未満なら最小ロットに設定
    if(lot > max_lot) lot = max_lot;  // 最大ロット超過なら最大ロットに設定

    return lot;
}

//+------------------------------------------------------------------+
//| 全ポジションをクローズする関数                                   |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
    int total = PositionsTotal();  // 現在のポジション数を取得

    for(int i = total - 1; i >= 0; i--)  // 逆順でポジションを処理
    {
        if(PositionGetSymbol(i) != NULL)  // インデックスからシンボル名を取得
        {
            string symbol = PositionGetSymbol(i);  // ポジションのシンボルを取得
            
            if(PositionSelect(symbol))  // ポジションを選択
            {
                // ポジションをクローズ
                if(trade.PositionClose(symbol))
                {
                    PrintFormat("ポジションをクローズしました: %s", symbol);
                }
                else
                {
                    PrintFormat("ポジションのクローズに失敗: %s, エラー: %d", symbol, GetLastError());
                }
            }
        }
    }
}
