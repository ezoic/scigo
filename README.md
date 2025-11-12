# SciGo 🚀

<div align="center">
  <img src="docs/GOpher.png" alt="SciGo Mascot Gopher" width="200"/>
  <p><i>SciGo's official mascot - Ready, Set, SciGo!</i></p>
  
  **The blazing-fast scikit-learn compatible ML library for Go**
  
  Say "Goodbye" to slow ML, "Sci-Go" to fast learning!
  
  [![CI](https://github.com/ezoic/scigo/actions/workflows/ci.yml/badge.svg)](https://github.com/ezoic/scigo/actions/workflows/ci.yml)
  [![Codecov](https://codecov.io/gh/YuminosukeSato/scigo/branch/main/graph/badge.svg)](https://codecov.io/gh/YuminosukeSato/scigo)
  [![Go Report Card](https://goreportcard.com/badge/github.com/ezoic/scigo)](https://goreportcard.com/report/github.com/ezoic/scigo)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Go Version](https://img.shields.io/badge/Go-1.23%2B-blue.svg)](https://go.dev/)
  [![GoDoc](https://pkg.go.dev/badge/github.com/ezoic/scigo)](https://pkg.go.dev/github.com/ezoic/scigo)
  [![Release](https://img.shields.io/github/v/release/YuminosukeSato/scigo)](https://github.com/ezoic/scigo/releases)
</div>

---

## 🌟 Why SciGo?

**SciGo** = **S**tatistical **C**omputing **I**n **Go**

SciGo brings the power and familiarity of scikit-learn to the Go ecosystem, offering:

- 🔥 **Blazing Fast**: Native Go implementation with built-in parallelization
- 🎯 **scikit-learn Compatible**: Familiar Fit/Predict API for easy migration
- 🌲 **LightGBM Support**: Full compatibility with Python LightGBM models (.txt/JSON/string)
- 📖 **Well Documented**: Complete API documentation with examples on [pkg.go.dev](https://pkg.go.dev/github.com/ezoic/scigo)
- 🌊 **Streaming Support**: Online learning algorithms for real-time data
- 🚀 **Zero Heavy Dependencies**: Pure Go implementation (only scientific essentials)
- 📊 **Comprehensive**: Regression, classification, clustering, tree-based models, and more
- 🧪 **Production Ready**: Extensive tests, benchmarks, and error handling
- ⚡ **Superior to leaves**: Not just inference - full training, convenience features, and numerical precision

## 📦 Installation

### Go Module (Recommended)
```bash
go get github.com/ezoic/scigo@latest
```

### Quick Start Options
- **🐳 Docker**: `docker run --rm -it ghcr.io/yuminosukesato/scigo:latest`
- **☁️ GitPod**: [![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/ezoic/scigo)
- **📦 Go Install**: `go install github.com/ezoic/scigo/examples/quick-start@latest`

## 🚀 Quick Start

> 💡 **Tip**: For complete API documentation with examples, visit [pkg.go.dev/scigo](https://pkg.go.dev/github.com/ezoic/scigo)

### Option 1: One-Liner with LightGBM 🌲
```go
package main

import (
    "github.com/ezoic/scigo/sklearn/lightgbm"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Super convenient one-liner training!
    X := mat.NewDense(100, 4, data) // Your data
    y := mat.NewDense(100, 1, labels) // Your labels
    
    // Train and predict in one line!
    result := lightgbm.QuickTrain(X, y)
    predictions := result.Predict(X_test)
    
    // Or use AutoML for automatic tuning
    best := lightgbm.AutoFit(X, y)
    
    // Load Python LightGBM models directly!
    model := lightgbm.NewLGBMClassifier()
    model.LoadModel("python_model.txt") // Full compatibility!
    predictions, _ := model.Predict(X_test)
}
```

### Option 2: Classic Linear Regression
```go
package main

import (
    "fmt"
    "log"
    
    "github.com/ezoic/scigo/linear"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Create and train model - just like scikit-learn!
    model := linear.NewLinearRegression()
    
    // Training data
    X := mat.NewDense(4, 2, []float64{
        1, 1,
        1, 2,
        2, 2,
        2, 3,
    })
    y := mat.NewDense(4, 1, []float64{
        2, 3, 3, 4,
    })
    
    // Fit the model
    if err := model.Fit(X, y); err != nil {
        log.Fatal(err)
    }
    
    // Make predictions
    XTest := mat.NewDense(2, 2, []float64{
        1.5, 1.5,
        2.5, 3.5,
    })
    predictions, _ := model.Predict(XTest)
    
    fmt.Println("Ready, Set, SciGo! Predictions:", predictions)
}
```

## 📚 API Documentation

[![GoDoc](https://pkg.go.dev/badge/github.com/ezoic/scigo)](https://pkg.go.dev/github.com/ezoic/scigo)

### 📖 Package Documentation

| Package | Description | Go Doc |
|---------|-------------|--------|
| **sklearn/lightgbm** 🌲 | LightGBM with Python model compatibility & convenience features | [![GoDoc](https://pkg.go.dev/badge/github.com/ezoic/scigo/sklearn/lightgbm)](https://pkg.go.dev/github.com/ezoic/scigo/sklearn/lightgbm) |
| **sklearn/linear_model** | Linear models with full scikit-learn compatibility | [![GoDoc](https://pkg.go.dev/badge/github.com/ezoic/scigo/sklearn/linear_model)](https://pkg.go.dev/github.com/ezoic/scigo/sklearn/linear_model) |
| **preprocessing** | Data preprocessing utilities (StandardScaler, MinMaxScaler, OneHotEncoder) | [![GoDoc](https://pkg.go.dev/badge/github.com/ezoic/scigo/preprocessing)](https://pkg.go.dev/github.com/ezoic/scigo/preprocessing) |
| **linear** | Linear machine learning algorithms (LinearRegression) | [![GoDoc](https://pkg.go.dev/badge/github.com/ezoic/scigo/linear)](https://pkg.go.dev/github.com/ezoic/scigo/linear) |
| **metrics** | Model evaluation metrics (MSE, RMSE, MAE, R², MAPE) | [![GoDoc](https://pkg.go.dev/badge/github.com/ezoic/scigo/metrics)](https://pkg.go.dev/github.com/ezoic/scigo/metrics) |
| **core/model** | Base model with weight export/import and scikit-learn compatibility | [![GoDoc](https://pkg.go.dev/badge/github.com/ezoic/scigo/core/model)](https://pkg.go.dev/github.com/ezoic/scigo/core/model) |

### 📋 Complete API Examples

The documentation includes comprehensive examples for all major APIs. Visit the Go Doc links above or use `go doc` locally:

```bash
# View package documentation
go doc github.com/ezoic/scigo/preprocessing
go doc github.com/ezoic/scigo/linear
go doc github.com/ezoic/scigo/metrics

# View specific function documentation
go doc github.com/ezoic/scigo/preprocessing.StandardScaler.Fit
go doc github.com/ezoic/scigo/linear.LinearRegression.Predict
go doc github.com/ezoic/scigo/metrics.MSE

# Run example tests
go test -v ./preprocessing -run Example
go test -v ./linear -run Example
go test -v ./metrics -run Example
```

## 📚 Algorithms

### Supervised Learning

#### Linear Models
- ✅ **Linear Regression** - Full scikit-learn compatible implementation with QR decomposition
- ✅ **SGD Regressor** - Stochastic Gradient Descent for large-scale learning
- ✅ **SGD Classifier** - Linear classifiers with SGD training
- ✅ **Passive-Aggressive** - Online learning for classification and regression

### Data Preprocessing
- ✅ **StandardScaler** - Standardizes features by removing mean and scaling to unit variance
- ✅ **MinMaxScaler** - Scales features to a given range (e.g., [0,1] or [-1,1])
- ✅ **OneHotEncoder** - Encodes categorical features as one-hot numeric arrays

#### Tree-based Models
- ✅ **LightGBM** - Full Python model compatibility (.txt/JSON/string formats)
  - LGBMClassifier - Binary and multiclass classification
  - LGBMRegressor - Regression with multiple objectives
  - QuickTrain - One-liner training with automatic model selection
  - AutoFit - Automatic hyperparameter tuning
  - Superior to [leaves](https://github.com/dmitryikh/leaves) - training + convenience features
- 🚧 Random Forest (Coming Soon)
- 🚧 XGBoost compatibility (Coming Soon)

### Unsupervised Learning

#### Clustering
- ✅ **MiniBatch K-Means** - Scalable K-Means for large datasets
- 🚧 DBSCAN (Coming Soon)
- 🚧 Hierarchical Clustering (Coming Soon)

### Special Features

#### Online Learning & Streaming
- ✅ **Incremental Learning** - Update models with new data batches
- ✅ **Partial Fit** - scikit-learn compatible online learning
- ✅ **Concept Drift Detection** - DDM and ADWIN algorithms
- ✅ **Streaming Pipelines** - Real-time data processing with channels

## 🎯 scikit-learn Compatibility

SciGo implements the familiar scikit-learn API with full compatibility:

```go
// Just like scikit-learn!
model.Fit(X, y)              // Train the model
model.Predict(X)              // Make predictions  
model.Score(X, y)             // Evaluate the model
model.PartialFit(X, y)        // Incremental learning

// New in v0.3.0 - Full scikit-learn compatibility
model.GetParams(deep)         // Get model parameters
model.SetParams(params)       // Set model parameters
weights, _ := model.ExportWeights()  // Export model weights
model.ImportWeights(weights)  // Import with guaranteed reproducibility

// Streaming - unique to Go!
model.FitStream(ctx, dataChan) // Streaming training
```

### 🆕 New Features in v0.3.0

- **Complete Weight Reproducibility** - Guaranteed identical outputs with same weights
- **gRPC/Protobuf Support** - Distributed training and prediction
- **Full Parameter Management** - GetParams/SetParams for all models
- **Model Serialization** - Export/Import with full precision

## 📊 Performance Benchmarks

SciGo leverages Go's concurrency for exceptional performance:

| Algorithm | Dataset Size | SciGo | scikit-learn (Python) | Speedup |
|-----------|-------------|-------|--------------------|---------|
| Linear Regression | 1M×100 | 245ms | 890ms | **3.6×** |
| SGD Classifier | 500K×50 | 180ms | 520ms | **2.9×** |
| MiniBatch K-Means | 100K×20 | 95ms | 310ms | **3.3×** |
| Streaming SGD | 1M streaming | 320ms | 1.2s | **3.8×** |

*Benchmarks on MacBook Pro M2, 16GB RAM*

### Memory Efficiency

| Dataset Size | Memory | Allocations |
|-------------|--------|-------------|
| 100×10 | 22.8KB | 22 |
| 1,000×10 | 191.8KB | 22 |
| 10,000×20 | 3.4MB | 57 |
| 50,000×50 | 41.2MB | 61 |

## 🏗️ Architecture

```
scigo/
├── linear/           # Linear models
├── sklearn/          # scikit-learn compatible implementations
│   ├── linear_model/ # SGD, Passive-Aggressive
│   ├── cluster/      # Clustering algorithms
│   └── drift/        # Concept drift detection
├── metrics/          # Evaluation metrics
├── core/            # Core abstractions
│   ├── model/       # Base model interfaces
│   ├── tensor/      # Tensor operations
│   └── parallel/    # Parallel processing
├── datasets/        # Dataset utilities
└── examples/        # Usage examples
```

## 📊 Metrics

Comprehensive evaluation metrics with full documentation:

- **Regression Metrics**: 
  - MSE (Mean Squared Error) - [`pkg.go.dev/metrics.MSE`](https://pkg.go.dev/github.com/ezoic/scigo/metrics#MSE)
  - RMSE (Root Mean Squared Error) - [`pkg.go.dev/metrics.RMSE`](https://pkg.go.dev/github.com/ezoic/scigo/metrics#RMSE)  
  - MAE (Mean Absolute Error) - [`pkg.go.dev/metrics.MAE`](https://pkg.go.dev/github.com/ezoic/scigo/metrics#MAE)
  - R² (Coefficient of Determination) - [`pkg.go.dev/metrics.R2Score`](https://pkg.go.dev/github.com/ezoic/scigo/metrics#R2Score)
  - MAPE (Mean Absolute Percentage Error) - [`pkg.go.dev/metrics.MAPE`](https://pkg.go.dev/github.com/ezoic/scigo/metrics#MAPE)
  - Explained Variance Score - [`pkg.go.dev/metrics.ExplainedVarianceScore`](https://pkg.go.dev/github.com/ezoic/scigo/metrics#ExplainedVarianceScore)
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC (coming)
- **Clustering**: Silhouette Score, Davies-Bouldin Index (coming)

## 🧪 Testing & Quality

```bash
# Run tests
go test ./...

# Run benchmarks
go test -bench=. -benchmem ./...

# Check coverage (76.7% overall coverage)
go test -cover ./...

# Run linter (errcheck, govet, ineffassign, staticcheck, unused, misspell)
make lint-full

# Run examples to see API usage
go test -v ./preprocessing -run Example
go test -v ./linear -run Example
go test -v ./metrics -run Example
go test -v ./core/model -run Example
```

### Quality Gates
- ✅ **Test Coverage**: 76.7% (target: 70%+)
- ✅ **Linting**: golangci-lint with comprehensive checks
- ✅ **Documentation**: Complete godoc for all public APIs
- ✅ **Examples**: Comprehensive example functions for all major APIs

## 📚 Examples

Check out the [examples](examples/) directory:

- [Linear Regression](examples/linear_regression/) - Basic regression
- [Streaming Learning](examples/streaming_demo/) - Online learning demo
- [Iris Classification](examples/iris_regression/) - Classic dataset
- [Error Handling](examples/error_demo/) - Robust error management

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ezoic/scigo.git
cd scigo

# Install dependencies
go mod download

# Run tests
go test ./...

# Run linter
golangci-lint run
```

### 🚀 Continuous Delivery (CD)

SciGo uses automated continuous delivery for releases:

- **Automatic Release**: Every push to the `main` branch triggers an automatic patch version release
- **Version Management**: Versions are automatically incremented (e.g., 0.4.0 → 0.4.1)
- **Release Assets**: Binaries for Linux, macOS, and Windows are automatically built and attached
- **Docker Images**: Docker images are automatically built and pushed to GitHub Container Registry (ghcr.io)
- **Documentation**: pkg.go.dev is automatically updated with the latest version

#### Release Process

1. **Merge PR to main**: When a PR is merged to main branch
2. **Automatic Tests**: CI runs all tests and coverage checks
3. **Version Bump**: Patch version is automatically incremented
4. **Create Release**: GitHub Release is created with:
   - Multi-platform binaries (Linux, macOS, Windows)
   - Release notes from CHANGELOG.md
   - Docker image at `ghcr.io/yuminosukesato/scigo:VERSION`
5. **Post-Release**: An issue is created to track post-release verification tasks

#### Manual Release

For major or minor version releases, create and push a tag manually:

```bash
git tag v0.5.0 -m "Release v0.5.0"
git push origin v0.5.0
```

This will trigger the release workflow via the existing `release.yml` workflow.

## 🗺️ Roadmap

### Phase 1: Core ML (Current)
- ✅ Linear models
- ✅ Online learning
- ✅ Basic clustering
- 🚧 Tree-based models

### Phase 2: Advanced Features
- [ ] Neural Networks (MLP)
- [ ] Deep Learning integration
- [ ] Model serialization (ONNX export)
- [ ] GPU acceleration

### Phase 3: Enterprise Features
- [ ] Distributed training
- [ ] AutoML capabilities
- [ ] Model versioning
- [ ] A/B testing framework

## 📖 Documentation

### Core Documentation
- **[API Documentation](https://pkg.go.dev/github.com/ezoic/scigo)** - Complete API reference with examples
- **[Package Index](https://pkg.go.dev/github.com/ezoic/scigo?tab=subdirectories)** - Browse all packages

### API Quick Reference
| API | Package | Documentation |
|-----|---------|---------------|
| `StandardScaler` | preprocessing | [pkg.go.dev/preprocessing.StandardScaler](https://pkg.go.dev/github.com/ezoic/scigo/preprocessing#StandardScaler) |
| `MinMaxScaler` | preprocessing | [pkg.go.dev/preprocessing.MinMaxScaler](https://pkg.go.dev/github.com/ezoic/scigo/preprocessing#MinMaxScaler) |
| `OneHotEncoder` | preprocessing | [pkg.go.dev/preprocessing.OneHotEncoder](https://pkg.go.dev/github.com/ezoic/scigo/preprocessing#OneHotEncoder) |
| `LinearRegression` | linear | [pkg.go.dev/linear.LinearRegression](https://pkg.go.dev/github.com/ezoic/scigo/linear#LinearRegression) |
| `BaseEstimator` | core/model | [pkg.go.dev/model.BaseEstimator](https://pkg.go.dev/github.com/ezoic/scigo/core/model#BaseEstimator) |

### Migration & Advanced Guides
- **[📚 scikit-learn Migration Guide](docs/sklearn-migration-guide.md)** - Complete guide for Python developers
- **[🏗️ API Stability Analysis](docs/api-stability.md)** - v1.0.0 roadmap and compatibility
- [🌊 Streaming Guide](docs/streaming.md) (Coming Soon)
- [⚡ Performance Tuning](docs/performance.md) (Coming Soon)

## 🙏 Acknowledgments

- Inspired by [scikit-learn](https://scikit-learn.org/)
- Built with [Gonum](https://www.gonum.org/)
- Error handling by [CockroachDB errors](https://github.com/cockroachdb/errors)

## 📄 License

SciGo is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 📧 Contact

- **Author**: Yuminosuke Sato
- **GitHub**: [@YuminosukeSato](https://github.com/YuminosukeSato)
- **Repository**: [https://github.com/ezoic/scigo](https://github.com/ezoic/scigo)
- **Issues**: [GitHub Issues](https://github.com/ezoic/scigo/issues)

---

<div align="center">
  <h3>🚀 Ready, Set, SciGo! 🚀</h3>
  <i>Where Science Meets Go - Say goodbye to slow ML!</i>
  <br><br>
  Made with ❤️ and lots of ☕ in Go
</div>
### Running scikit-learn parity tests

Development-only parity tests compare the Go implementation against scikit-learn outputs.
They are not part of the default `go test`; use the `parity` build tag explicitly.

Steps

1. Generate golden data
   - Use `uv` instead of pip.
   - Command: `uv run --with scikit-learn --with numpy --with scipy python scripts/golden/gen_logreg.py`
2. Run parity tests
   - Command: `go test ./sklearn/linear_model -tags=parity -run Parity -v`

One-liner

`make parity-linear`

Notes
- Current LogisticRegression uses simplified gradient descent. After implementing lbfgs/newton-cg, tolerances will be tightened.
- Golden file is written to `tests/golden/logreg_case1.json`.
